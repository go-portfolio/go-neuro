package nn

type Trainer interface {
	Update(net *Network, samples, targets [][]float64, epochs int)
}

type Network struct {
	Layers []*Layer
}

func NewNetwork(sizes []int) *Network {
	layers := make([]*Layer, 0)
	for i := 0; i < len(sizes)-1; i++ {
		layers = append(layers, NewLayer(sizes[i], sizes[i+1]))
	}
	return &Network{Layers: layers}
}

func (net *Network) Feedforward(x []float64) []float64 {
	in := x
	for _, L := range net.Layers {
		in = L.Forward(in)
	}
	return in
}

func (net *Network) Fit(t Trainer, samples, targets [][]float64, epochs int) {
	t.Update(net, samples, targets, epochs)
}

func (net *Network) Predict(x []float64) []float64 {
	return net.Feedforward(x)
}
