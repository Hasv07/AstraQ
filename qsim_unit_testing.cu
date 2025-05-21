#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <math.h>    

typedef struct {
    float real;
    float imag;
} Complex;

extern "C" {
    void* create_state(int num_qubits);
    void apply_gate(void* state, const char* gate, int qubit, int target, int num_qubits);
    void measure(void* state, int* results, int num_shots, int num_qubits);
} 

const float EPSILON = 1e-5;

// Utility function to copy device state to host
void copy_state_to_host(void* d_state, Complex* h_state, int size) {
    cudaMemcpy(h_state, d_state, size * sizeof(Complex), cudaMemcpyDeviceToHost);
}

TEST(QuantumSimulatorTest, InitialState) {
    int num_qubits = 1;
    void* state = create_state(num_qubits);
    int size = 1 << num_qubits;
    Complex* h_state = new Complex[size];
    copy_state_to_host(state, h_state, size);

    EXPECT_NEAR(h_state[0].real, 1.0f, EPSILON);
    EXPECT_NEAR(h_state[0].imag, 0.0f, EPSILON);
    for (int i = 1; i < size; ++i) {
        EXPECT_NEAR(h_state[i].real, 0.0f, EPSILON);
        EXPECT_NEAR(h_state[i].imag, 0.0f, EPSILON);
    }

    delete[] h_state;
    cudaFree(state);
}

TEST(QuantumSimulatorTest, HadamardGate) {
    int num_qubits = 1;
    void* state = create_state(num_qubits);
    apply_gate(state, "h", 0, -1, num_qubits);
    int size = 1 << num_qubits;
    Complex* h_state = new Complex[size];
    copy_state_to_host(state, h_state, size);

    float expected = 1.0f / sqrtf(2.0f);
    for (int i = 0; i < size; ++i) {
        EXPECT_NEAR(h_state[i].real, expected, EPSILON);
        EXPECT_NEAR(h_state[i].imag, 0.0f, EPSILON);
    }

    delete[] h_state;
    cudaFree(state);
}

TEST(QuantumSimulatorTest, PauliXGate) {
    int num_qubits = 1;
    void* state = create_state(num_qubits);
    apply_gate(state, "x", 0, -1, num_qubits);
    int size = 1 << num_qubits;
    Complex* h_state = new Complex[size];
    copy_state_to_host(state, h_state, size);

    EXPECT_NEAR(h_state[0].real, 0.0f, EPSILON);
    EXPECT_NEAR(h_state[0].imag, 0.0f, EPSILON);
    EXPECT_NEAR(h_state[1].real, 1.0f, EPSILON);
    EXPECT_NEAR(h_state[1].imag, 0.0f, EPSILON);

    delete[] h_state;
    cudaFree(state);
}

TEST(QuantumSimulatorTest, CNOTBellState) {
    int num_qubits = 2;
    void* state = create_state(num_qubits);
    apply_gate(state, "h", 0, -1, num_qubits);
    apply_gate(state, "cx", 0, 1, num_qubits);
    int size = 1 << num_qubits;
    Complex* h_state = new Complex[size];
    copy_state_to_host(state, h_state, size);

    float expected = 1.0f / sqrtf(2.0f);
    EXPECT_NEAR(h_state[0].real, expected, EPSILON);
    EXPECT_NEAR(h_state[0].imag, 0.0f, EPSILON);
    EXPECT_NEAR(h_state[1].real, 0.0f, EPSILON);
    EXPECT_NEAR(h_state[1].imag, 0.0f, EPSILON);
    EXPECT_NEAR(h_state[2].real, 0.0f, EPSILON);
    EXPECT_NEAR(h_state[2].imag, 0.0f, EPSILON);
    EXPECT_NEAR(h_state[3].real, expected, EPSILON);
    EXPECT_NEAR(h_state[3].imag, 0.0f, EPSILON);

    delete[] h_state;
    cudaFree(state);
}

TEST(QuantumSimulatorTest, MeasurementDeterministic) {
    int num_qubits = 1;
    void* state = create_state(num_qubits);
    int num_shots = 1000;
    int* results = new int[num_shots];
    measure(state, results, num_shots, num_qubits);

    for (int i = 0; i < num_shots; ++i) {
        EXPECT_EQ(results[i], 0);
    }

    delete[] results;
    cudaFree(state);
}

TEST(QuantumSimulatorTest, PauliYGate) {
    int num_qubits = 1;
    void* state = create_state(num_qubits);
    apply_gate(state, "y", 0, -1, num_qubits);
    int size = 1 << num_qubits;
    Complex* h_state = new Complex[size];
    copy_state_to_host(state, h_state, size);

    EXPECT_NEAR(h_state[0].real, 0.0f, EPSILON);
    EXPECT_NEAR(h_state[0].imag, 0.0f, EPSILON);
    EXPECT_NEAR(h_state[1].real, 0.0f, EPSILON);
    EXPECT_NEAR(h_state[1].imag, 1.0f, EPSILON);

    delete[] h_state;
    cudaFree(state);
}

TEST(QuantumSimulatorTest, PauliZGate) {
    int num_qubits = 1;
    void* state = create_state(num_qubits);
    apply_gate(state, "h", 0, -1, num_qubits);
    apply_gate(state, "z", 0, -1, num_qubits);
    int size = 1 << num_qubits;
    Complex* h_state = new Complex[size];
    copy_state_to_host(state, h_state, size);

    float expected = 1.0f / sqrtf(2.0f);
    EXPECT_NEAR(h_state[0].real, expected, EPSILON);
    EXPECT_NEAR(h_state[0].imag, 0.0f, EPSILON);
    EXPECT_NEAR(h_state[1].real, -expected, EPSILON);
    EXPECT_NEAR(h_state[1].imag, 0.0f, EPSILON);

    delete[] h_state;
    cudaFree(state);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}