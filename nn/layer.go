package nn

type Layer struct {
	In      int
	Out     int
	Weights [][]float64
	Biases  []float64
	Z       []float64
	A       []float64
}

func NewLayer(in, out int) *Layer {
	L := &Layer{
		In:      in,
		Out:     out,
		Weights: make([][]float64, out),
		Biases:  make([]float64, out),
		Z:       make([]float64, out),
		A:       make([]float64, out),
	}
	for i := 0; i < out; i++ {
		L.Weights[i] = make([]float64, in)
		for j := 0; j < in; j++ {
			L.Weights[i][j] = RandomFloat()
		}
		L.Biases[i] = RandomFloat()
	}
	return L
}

func (L *Layer) Forward(x []float64) []float64 {
	for i := 0; i < L.Out; i++ {
		sum := L.Biases[i]
		for j := 0; j < L.In; j++ {
			sum += L.Weights[i][j] * x[j]
		}
		L.Z[i] = sum
		L.A[i] = Sigmoid(sum)
	}
	return append([]float64{}, L.A...)
}
