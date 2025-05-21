#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

extern "C" {
    void* create_state(int num_qubits);
    void apply_gate(void* state, const char* gate, int qubit, int target, int num_qubits);
    void measure(void* state, int* results, int num_shots, int num_qubits);
}

class QuantumSimulator {
public:
    QuantumSimulator(int num_qubits) : num_qubits(num_qubits) {
        state = create_state(num_qubits);
    }
    
    void h(int qubit) { apply_gate(state, "h", qubit, -1, num_qubits); }
    void cx(int control, int target) { apply_gate(state, "cx", control, target, num_qubits); }
    void x(int target) { apply_gate(state, "x", target, -1, num_qubits); } 
    void y(int target) { apply_gate(state, "y", target, -1, num_qubits); }
    void z(int target) { apply_gate(state, "z", target, -1, num_qubits); }
    void s(int target) { apply_gate(state, "s", target, -1, num_qubits); }
    void sdg(int target) { apply_gate(state, "sdg", target, -1, num_qubits); }
    void t(int target) { apply_gate(state, "t", target, -1, num_qubits); }
    void tdg(int target) { apply_gate(state, "tdg", target, -1, num_qubits); }
    void cz(int control, int target) { apply_gate(state, "cz", control, target, num_qubits); }
    void swap(int a, int b) { apply_gate(state, "swap", a, b, num_qubits); }

    std::vector<int> measure_all(int num_shots) {
        std::vector<int> results(num_shots);
        measure(state, results.data(), num_shots, num_qubits);
        return results;
    }
    int get_num_qubits() const { return num_qubits; }

private:
    void* state;
    int num_qubits;
};

PYBIND11_MODULE(qsim, m) {
    py::class_<QuantumSimulator>(m, "QuantumCircuit")
        .def(py::init<int>())
        .def("num_qubits", &QuantumSimulator::get_num_qubits)
        .def("h", &QuantumSimulator::h)
        .def("cx", &QuantumSimulator::cx)
        .def("x", &QuantumSimulator::x)
        .def("y", &QuantumSimulator::y)
        .def("z", &QuantumSimulator::z)
        .def("s", &QuantumSimulator::s)
        .def("sdg", &QuantumSimulator::sdg)
        .def("t", &QuantumSimulator::t)
        .def("tdg", &QuantumSimulator::tdg)
        .def("cz", &QuantumSimulator::cz)
        .def("swap", &QuantumSimulator::swap)
        .def("measure_all", &QuantumSimulator::measure_all);
}