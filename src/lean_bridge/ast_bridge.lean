-- TODO(kernel-expr-ir | tier:A | blocks:[ast-admissibility-gate, tree-rotor-operator]):
-- Description: Extract Lean 4 kernel Lean.Expr 11-constructor DAG (bvar, fvar, mvar, sort, const, app, lam, forallE, letE, lit, proj) directly without text BPE tokenization.
-- Criterion: assert: expr_dag_valid and bpe_tokens_used == 0
import Lean
open Lean

partial def syntaxToPreorder (s : Syntax) : List String :=
  match s with
  | Syntax.node _ k args =>
    if k == `hygieneInfo then []
    else
      let children := args.toList.flatMap syntaxToPreorder
      if k == `null || k == `Lean.Parser.Term.hygienicLParen then children else s!"NODE:{k}" :: children
  | Syntax.ident _ _ val _ =>
    if val.isAnonymous then [] else [s!"IDENT:{val}"]
  | Syntax.atom _ val =>
    if val == "←" then ["ATOM:←"] else []
  | Syntax.missing => []

partial def mapSyntaxIdents (s : Syntax) (f : String → String) : Syntax :=
  match s with
  | Syntax.node info k args =>
    if k == `hygieneInfo then s
    else Syntax.node info k (args.map (mapSyntaxIdents · f))
  | Syntax.ident info _ val preres =>
    if val.isAnonymous then s
    else
      let full := val.toString
      let mapped :=
        let direct := f full
        if direct != full then direct
        else ".".intercalate ((full.splitOn ".").map f)
      Syntax.ident info mapped.toRawSubstring mapped.toName preres
  | other => other

partial def collectBinderDecls (s : Syntax) : List (String × String) :=
  match s with
  | Syntax.node _ `Lean.Parser.Term.explicitBinder args
  | Syntax.node _ `Lean.Parser.Term.implicitBinder args =>
    let ids := if _ : 1 < args.size then
      args[1]!.getArgs.toList.filterMap (fun idStx =>
        match idStx with
        | Syntax.ident _ _ val _ => some val.toString
        | _ => none)
    else []
    let tyStr := if _ : 2 < args.size then
      (args[2]!.reprint.getD "").trimAscii.toString
    else ""
    ids.map (·, tyStr)
  | Syntax.node _ _ args => args.toList.flatMap collectBinderDecls
  | _ => []

def lookupMap (pairs : List (String × String)) (k : String) : String :=
  match pairs.find? (fun p => p.1 == k) with
  | some (_, v) => v
  | none => k

def jsonPairsToMap (j : Json) : List (String × String) :=
  match j.getObj? with
  | Except.ok obj =>
    obj.toList.filterMap (fun (k, v) =>
      match v.getStr? with
      | Except.ok s => some (k, s)
      | Except.error _ => none)
  | Except.error _ => []

unsafe def handleRequest (env : Environment) (req : Json) : Json := Id.run do
  let op := (req.getObjValAs? String "op").toOption.getD ""
  if op == "parse_tactic" then
    let tac := (req.getObjValAs? String "tactic").toOption.getD ""
    let sigma := match req.getObjVal? "sigma" with
      | Except.ok j => jsonPairsToMap j
      | Except.error _ => []
    let parts := tac.splitOn ";"
    let mut preAll : List String := []
    let mut reprintedParts : List String := []
    let mut rootKind : String := ""
    for p in parts do
      let pTrim := p.trimAscii.toString
      if !pTrim.isEmpty then
        match Parser.runParserCategory env `tactic pTrim with
        | Except.error _ =>
          reprintedParts := reprintedParts ++ [pTrim]
        | Except.ok stx =>
          let stx' := if sigma.isEmpty then stx else mapSyntaxIdents stx (lookupMap sigma)
          if rootKind.isEmpty then
            rootKind := stx'.getKind.toString
          preAll := preAll ++ syntaxToPreorder stx'
          let rep := (stx'.reprint.getD pTrim).trimAscii.toString
          reprintedParts := reprintedParts ++ [rep]
    Json.mkObj [
      ("ok", Json.bool true),
      ("root_kind", Json.str rootKind),
      ("preorder", toJson preAll),
      ("canonical_tactic", Json.str ("; ".intercalate reprintedParts))
    ]
  else if op == "canonicalize_term" then
    let termStr := (req.getObjValAs? String "term").toOption.getD ""
    let sigma := match req.getObjVal? "sigma" with
      | Except.ok j => jsonPairsToMap j
      | Except.error _ => []
    match Parser.runParserCategory env `term termStr with
    | Except.error _ =>
      Json.mkObj [("ok", Json.bool false), ("canonical_term", Json.str termStr)]
    | Except.ok stx =>
      let stx' := if sigma.isEmpty then stx else mapSyntaxIdents stx (lookupMap sigma)
      let rep := (stx'.reprint.getD termStr).trimAscii.toString
      Json.mkObj [("ok", Json.bool true), ("canonical_term", Json.str rep)]
  else if op == "parse_command_binders" then
    let cmd := (req.getObjValAs? String "command").toOption.getD ""
    let hasBy := cmd.endsWith "by"
    let fullCmd := if hasBy then cmd ++ " sorry" else cmd
    match Parser.runParserCategory env `command fullCmd with
    | Except.error e =>
      Json.mkObj [("ok", Json.bool false), ("error", Json.str e)]
    | Except.ok stx =>
      let binders := collectBinderDecls stx
      let binderJson := binders.map (fun (id, ty) => Json.mkObj [("ident", Json.str id), ("type", Json.str ty)])
      Json.mkObj [
        ("ok", Json.bool true),
        ("kind", Json.str stx.getKind.toString),
        ("binders", toJson binderJson)
      ]
  else
    Json.mkObj [("ok", Json.bool false), ("error", Json.str s!"unknown op: {op}")]

partial def loopLines (stdin : IO.FS.Stream) (stdout : IO.FS.Stream) (f : String → String) : IO Unit := do
  let line ← stdin.getLine
  if line.isEmpty then
    return ()
  let trimmed := line.trimAscii.toString
  if !trimmed.isEmpty then
    stdout.putStrLn (f trimmed)
    stdout.flush
  loopLines stdin stdout f

unsafe def main : IO Unit := do
  initSearchPath (← findSysroot)
  enableInitializersExecution
  let env ← importModules #[{ module := `Init }] {} (trustLevel := 0) (loadExts := true)
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  loopLines stdin stdout (fun line =>
    match Json.parse line with
    | Except.error e => (Json.mkObj [("ok", Json.bool false), ("error", Json.str e)]).compress
    | Except.ok req => (handleRequest env req).compress
  )
