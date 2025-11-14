package nn

import (
	"math"
	"math/rand"
)

// коэффициент утечки
const leakyAlpha = 0.01

// ========================
// Структура слоя нейросети
// ========================
type Layer struct {
	In          int
	Out         int
	Weights     [][]float64
	Biases      []float64
	Z           []float64
	A           []float64
	DropoutProb float64
}

// ========================
// Создание нового слоя (Xavier Init)
// ========================
func NewLayer(in, out int) *Layer {
	L := &Layer{
		In:          in,
		Out:         out,
		Weights:     make([][]float64, out),
		Biases:      make([]float64, out),
		Z:           make([]float64, out),
		A:           make([]float64, out),
		DropoutProb: 0.0,
	}

	// Xavier normal init
	std := math.Sqrt(2.0 / float64(in+out))

	for i := 0; i < out; i++ {
		L.Weights[i] = make([]float64, in)
		L.Biases[i] = 0.0

		for j := 0; j < in; j++ {
			L.Weights[i][j] = rand.NormFloat64() * std
		}
	}

	return L
}

// ========================
// Leaky ReLU
// ========================
func leakyReLU(x float64) float64 {
	if x > 0 {
		return x
	}
	return leakyAlpha * x
}

func leakyReLUPrime(x float64) float64 {
	if x > 0 {
		return 1
	}
	return leakyAlpha
}

// ========================
// Forward pass
// ========================
func (L *Layer) Forward(x []float64) []float64 {
	for i := 0; i < L.Out; i++ {
		sum := L.Biases[i]
		for j := 0; j < L.In; j++ {
			sum += L.Weights[i][j] * x[j]
		}
		L.Z[i] = sum
		L.A[i] = leakyReLU(sum)
	}

	out := make([]float64, L.Out)
	copy(out, L.A)
	return out
}
