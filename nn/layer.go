package nn

import (
	"math"
	"math/rand"
)

// ========================
// Структура слоя нейросети
// ========================
type Layer struct {
	In          int         // количество входов
	Out         int         // количество нейронов
	Weights     [][]float64 // матрица весов: [Out][In]
	Biases      []float64   // смещения
	Z           []float64   // z = w*x + b
	A           []float64   // активации после ReLU
	DropoutProb float64     // вероятность dropout
}

// ========================
// Создание нового слоя (Xavier init)
// ========================
func NewLayer(in, out int) *Layer {
	L := &Layer{
		In:          in,
		Out:         out,
		Weights:     make([][]float64, out),
		Biases:      make([]float64, out),
		Z:           make([]float64, out),
		A:           make([]float64, out),
		DropoutProb: 0.0, // по умолчанию dropout отключён (сеть сама задаёт)
	}

	// Xavier normal init: std = sqrt(2 / (fan_in + fan_out))
	std := math.Sqrt(2.0 / float64(in+out))

	for i := 0; i < out; i++ {
		L.Weights[i] = make([]float64, in)
		L.Biases[i] = 0.0 // как принято для ReLU

		for j := 0; j < in; j++ {
			L.Weights[i][j] = rand.NormFloat64() * std
		}
	}

	return L
}


// ========================
// Прямой проход (forward pass) слоя
// ========================
func (L *Layer) Forward(x []float64) []float64 {
	for i := 0; i < L.Out; i++ {
		sum := L.Biases[i]
		for j := 0; j < L.In; j++ {
			sum += L.Weights[i][j] * x[j]
		}
		L.Z[i] = sum
		L.A[i] = relu(sum)
	}

	// Возвращаем копию массива
	out := make([]float64, L.Out)
	copy(out, L.A)
	return out
}
