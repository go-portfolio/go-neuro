package nn

type Layer struct {
	in      int
	out     int
	weights [][]float64
	biases  []float64
	zs      [][]float64
	as      [][]float64
}

func NewLayer(in, out int) *Layer {
	L := &Layer{
		in:      in,
		out:     out,
		weights: make([][]float64, out),
		biases:  make([]float64, out),
		zs:      make([][]float64, 0),
		as:      make([][]float64, 0),
	}

	for i := 0; i < out; i++ {
		L.weights[i] = make([]float64, in)
		for j := 0; j < in; j++ {
			L.weights[i][j] = randomFloat()
		}
		L.biases[i] = randomFloat()
	}

	return L
}

func (L *Layer) Forward(x []float64) []float64 {
	z := make([]float64, L.out)
	a := make([]float64, L.out)

	for i := 0; i < L.out; i++ {
		sum := L.biases[i]
		for j := 0; j < L.in; j++ {
			sum += L.weights[i][j] * x[j]
		}
		z[i] = sum
		a[i] = sigmoid(sum)
	}

	L.zs = [][]float64{z}
	L.as = [][]float64{a}

	return a
}
