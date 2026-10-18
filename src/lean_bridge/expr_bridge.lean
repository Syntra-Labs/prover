/-
Kernel `Lean.Expr` DAG Bridge (`kernel-expr-ir`).
Elaborates goal types with Lean 4's real elaborator (`Term.elabType` + `MVarId.intros`)
and emits the hash-consed kernel term DAG over the 11 `Lean.Expr` constructors as JSON.
No surface text and no BPE tokenization reach the model:
  * bound variables are de Bruijn indices (`bvar`), so binder names are erased;
  * local hypotheses are labelled by their position in the `LocalContext` (`fv0, fv1, ...`);
  * `mdata` annotations are stripped (they carry no logical content).
Protocol: one JSON request per stdin line, one JSON response per stdout line.
  {"op":"state_dag","type":"∀ (p : Prop) (h : p), p","names":["And.intro"]}
  {"op":"state_dag","type":"∀ (n : Nat), n + 0 = n","names":"*"}   -- "*" = whole env
-/
import Lean
open Lean Elab Meta

/-- Constructor ids: the fixed 11-symbol kernel alphabet. -/
def ctorId : Expr → Nat
  | .bvar ..    => 0 | .fvar ..    => 1 | .mvar ..  => 2 | .sort ..  => 3
  | .const ..   => 4 | .app ..     => 5 | .lam ..   => 6 | .forallE .. => 7
  | .letE ..    => 8 | .lit ..     => 9 | .proj ..  => 10 | .mdata .. => 11

structure DagState where
  nodes : Array Json := #[]
  cache : Std.HashMap Expr Nat := {}

abbrev DagM := StateT DagState MetaM

/-- Hash-consed post-order encoding: children always precede parents. -/
partial def encode (fv : FVarId → Option Nat) (e : Expr) : DagM Nat := do
  if let .mdata _ inner := e then return ← encode fv inner
  if let some i := (← get).cache[e]? then return i
  let (label, children) ← match e with
    | .bvar i          => pure (toString i, #[])
    | .fvar id         => pure (match fv id with
                                | some i => s!"fv{i}"
                                | none   => id.name.toString, #[])
    | .mvar id         => pure (id.name.toString, #[])
    | .sort u          => pure (toString (format u), #[])
    | .const n _       => pure (n.toString, #[])
    | .app f a         => do pure ("", #[← encode fv f, ← encode fv a])
    | .lam _ t b _     => do pure ("", #[← encode fv t, ← encode fv b])
    | .forallE _ t b _ => do pure ("", #[← encode fv t, ← encode fv b])
    | .letE _ t v b _  => do pure ("", #[← encode fv t, ← encode fv v, ← encode fv b])
    | .lit (.natVal v) => pure (toString v, #[])
    | .lit (.strVal s) => pure (s, #[])
    | .proj s i x      => do pure (s!"{s}.{i}", #[← encode fv x])
    | .mdata _ x       => do pure ("", #[← encode fv x])
  let id := (← get).nodes.size
  let node := Json.mkObj [("k", toJson (ctorId e)), ("l", Json.str label), ("c", toJson children)]
  modify fun s => { s with nodes := s.nodes.push node, cache := s.cache.insert e id }
  return id

def runMeta {α} (env : Environment) (x : MetaM α) : IO α := do
  let ctx : Core.Context := { fileName := "<expr_bridge>", fileMap := default, maxHeartbeats := 200000 * 1000 }
  let st : Core.State := { env }
  let (a, _) ← (x.run' {} {}).toIO ctx st
  return a

/-- Every public theorem / inductive constructor in the environment (the lemma universe). -/
def lemmaUniverse (env : Environment) : Array Name :=
  env.constants.fold (init := #[]) fun acc n ci =>
    let keep := match ci with
      | .thmInfo _ | .ctorInfo _ => true
      | _ => false
    if keep && !n.isInternalDetail && !isPrivateName n then acc.push n else acc

/-- Elaborate a closed goal type and introduce exactly `nIntros` leading binders
(`nIntros < 0` = all), reproducing the tactic state Lean shows right after `:= by`. -/
def withGoal {α} (env : Environment) (tyStr : String) (nIntros : Int)
    (k : Array FVarId → MVarId → MetaM α) : IO α := do
  let stx ← IO.ofExcept (Parser.runParserCategory env `term tyStr)
  runMeta env do
    let ty ← (Term.elabType stx <* Term.synthesizeSyntheticMVarsNoPostponing).run'
    let ty ← instantiateMVars ty
    let goal ← mkFreshExprMVar ty
    let (fvars, g) ← if nIntros < 0 then goal.mvarId!.intros
                     else goal.mvarId!.introN nIntros.toNat
    g.withContext (k fvars g)

/-- Encode the turnstile goal together with the kernel types of the candidate lemmas into
ONE shared DAG, so that structural equality of any two subterms is equality of node ids. -/
def stateDag (env : Environment) (tyStr : String) (nIntros : Int) (names : Array Name) : IO Json :=
  withGoal env tyStr nIntros fun fvars g => do
    let fv : FVarId → Option Nat := fun id => fvars.findIdx? (· == id)
    let act : DagM (Array Json × Nat × Array Json) := do
      let mut hyps := #[]
      for i in [:fvars.size] do
        let d ← fvars[i]!.getDecl
        let r ← encode fv (← instantiateMVars d.type)
        hyps := hyps.push (Json.mkObj [("fv", Json.str s!"fv{i}"),
                                       ("user", Json.str d.userName.eraseMacroScopes.toString),
                                       ("isProp", Json.bool (← isProp d.type)),
                                       ("type", toJson r)])
      let t ← encode fv (← instantiateMVars (← g.getType))
      let mut consts := #[]
      for n in names do
        if let some ci := env.find? n then
          consts := consts.push (Json.mkObj [("name", Json.str n.toString),
                                             ("type", toJson (← encode (fun _ => none) ci.type))])
      return (hyps, t, consts)
    let ((hyps, t, consts), st) ← act.run {}
    return Json.mkObj [("ok", Json.bool true), ("nodes", toJson st.nodes),
                       ("hyps", toJson hyps), ("target", toJson t), ("consts", toJson consts)]

/-- Tier-2 gate: which lemmas does Lean's own `apply` accept on this goal (unification up to
definitional unfolding)? State is rolled back after every attempt. -/
def applyCheck (env : Environment) (tyStr : String) (nIntros : Int) (names : Array Name) : IO Json :=
  withGoal env tyStr nIntros fun _ g => do
    let mut ok := #[]
    for n in names do
      let saved ← saveState
      try
        let c ← mkConstWithFreshMVarLevels n
        let _ ← withTransparency .default (g.apply c)
        ok := ok.push (Json.str n.toString)
      catch _ => pure ()
      saved.restore
    return Json.mkObj [("ok", Json.bool true), ("accepted", toJson ok)]

def handle (env : Environment) (req : Json) : IO Json := do
  let op := (req.getObjValAs? String "op").toOption.getD ""
  let ty := (req.getObjValAs? String "type").toOption.getD ""
  let nIntros : Int := (req.getObjValAs? Int "intros").toOption.getD (-1)
  let names : Array Name := match req.getObjValAs? String "names" with
    | .ok "*" => lemmaUniverse env
    | _ => ((req.getObjValAs? (Array String) "names").toOption.getD #[]).map (·.toName)
  try
    if op == "state_dag" then stateDag env ty nIntros names
    else if op == "apply_check" then applyCheck env ty nIntros names
    else
      return Json.mkObj [("ok", Json.bool false), ("error", Json.str s!"unknown op: {op}")]
  catch e =>
    return Json.mkObj [("ok", Json.bool false), ("error", Json.str (toString e))]
partial def loop (env : Environment) (stdin stdout : IO.FS.Stream) : IO Unit := do
  let line ← stdin.getLine
  if line.isEmpty then return
  let t := line.trimAscii.toString
  if !t.isEmpty then
    let resp ← match Json.parse t with
      | .error e => pure (Json.mkObj [("ok", Json.bool false), ("error", Json.str e)])
      | .ok req  => handle env req
    stdout.putStrLn resp.compress
    stdout.flush
  loop env stdin stdout

unsafe def main : IO Unit := do
  initSearchPath (← findSysroot)
  enableInitializersExecution
  let env ← importModules #[{ module := `Init }] {} (trustLevel := 0) (loadExts := true)
  loop env (← IO.getStdin) (← IO.getStdout)
