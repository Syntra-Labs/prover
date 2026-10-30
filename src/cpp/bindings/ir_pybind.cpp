#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kernel_expr.hpp"
#include "tactic_ast.hpp"

namespace py = pybind11;
using namespace syn::ir;

PYBIND11_MODULE(syn_ir_cpp, m) {
    m.doc() = "C++ Lean 4 Kernel Expr and Tactic AST engine bindings";

    // ExprKind Enum
    py::enum_<ExprKind>(m, "ExprKind")
        .value("BVAR", ExprKind::BVAR)
        .value("FVAR", ExprKind::FVAR)
        .value("MVAR", ExprKind::MVAR)
        .value("SORT", ExprKind::SORT)
        .value("CONST", ExprKind::CONST)
        .value("APP", ExprKind::APP)
        .value("LAM", ExprKind::LAM)
        .value("FORALLE", ExprKind::FORALLE)
        .value("LETE", ExprKind::LETE)
        .value("LIT", ExprKind::LIT)
        .value("PROJ", ExprKind::PROJ)
        .export_values();

    // Node
    py::class_<Node>(m, "Node")
        .def(py::init<uint8_t, std::string, std::vector<int>>(),
             py::arg("k"), py::arg("label"), py::arg("children") = std::vector<int>{})
        .def_readwrite("k", &Node::k)
        .def_readwrite("label", &Node::label)
        .def_readwrite("children", &Node::children);

    // Hyp
    py::class_<Hyp>(m, "Hyp")
        .def(py::init<std::string, std::string, int, bool>(),
             py::arg("fv"), py::arg("user"), py::arg("type"), py::arg("is_prop"))
        .def_readwrite("fv", &Hyp::fv)
        .def_readwrite("user", &Hyp::user)
        .def_readwrite("type", &Hyp::type)
        .def_readwrite("is_prop", &Hyp::is_prop);

    // KernelState
    py::class_<KernelState>(m, "KernelState")
        .def(py::init<std::vector<Node>, std::vector<Hyp>, int, std::unordered_map<std::string, int>>(),
             py::arg("nodes"), py::arg("hyps"), py::arg("target"),
             py::arg("consts") = std::unordered_map<std::string, int>{})
        .def_readwrite("nodes", &KernelState::nodes)
        .def_readwrite("hyps", &KernelState::hyps)
        .def_readwrite("target", &KernelState::target)
        .def_readwrite("consts", &KernelState::consts)
        .def("subterms", &KernelState::subterms, py::arg("root"))
        .def("head", &KernelState::head, py::arg("n"))
        .def("render", &KernelState::render, py::arg("n"));

    // Function: decl_to_pi_type
    m.def("decl_to_pi_type", &decl_to_pi_type, py::arg("decl"),
          "Converts theorem declaration into closed Pi type and binder count.");

    // NodeSort Enum
    py::enum_<NodeSort>(m, "NodeSort")
        .value("OP", NodeSort::OP)
        .value("DIR", NodeSort::DIR)
        .value("LOC", NodeSort::LOC)
        .value("HYP", NodeSort::HYP)
        .value("VAR", NodeSort::VAR)
        .value("TERM", NodeSort::TERM)
        .export_values();

    // ASTNode
    py::class_<ASTNode>(m, "ASTNode")
        .def(py::init<std::string, NodeSort, std::vector<NodeSort>>(),
             py::arg("token"), py::arg("sort"), py::arg("child_sorts") = std::vector<NodeSort>{})
        .def_readwrite("token", &ASTNode::token)
        .def_readwrite("sort", &ASTNode::sort)
        .def_readwrite("child_sorts", &ASTNode::child_sorts)
        .def("__repr__", [](const ASTNode& n) {
            return "ASTNode('" + n.token + "', sort=" + std::string(node_sort_name(n.sort)) + ")";
        });

    // Functions for AST
    m.def("valid_ast_candidates", &valid_ast_candidates,
          py::arg("required_sort"), py::arg("sigma"),
          py::arg("premises") = std::vector<std::string>{},
          py::arg("allow_combinators") = true);

    m.def("lower_ast_to_lean", &lower_ast_to_lean,
          py::arg("nodes"), py::arg("sigma") = std::unordered_map<std::string, std::string>{});
}
