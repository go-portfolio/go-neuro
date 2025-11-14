package nn

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
	Velocities  [][]float64 // для Momentum весов
	BiasVel     []float64   // для Momentum смещений
}

// ========================
// Создание нового слоя (Xavier Init)
// ========================
func NewLayer(in, out int) *Layer {
	w := make([][]float64, out)
	v := make([][]float64, out)
	for i := range w {
		w[i] = make([]float64, in)
		v[i] = make([]float64, in) // инициализация Velocity
		// здесь можно случайно заполнить веса
	}
	return &Layer{
		In:         in,
		Out:        out,
		Weights:    w,
		Biases:     make([]float64, out),
		Velocities: v,
		BiasVel:    make([]float64, out),
	}
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
