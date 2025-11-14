package nn

import "fmt"

type Network struct {
	layers []*Layer
}

func NewNetwork(sizes []int) *Network {
	layers := make([]*Layer, 0)
	for i := 0; i < len(sizes)-1; i++ {
		layers = append(layers, NewLayer(sizes[i], sizes[i+1]))
	}
	return &Network{layers: layers}
}

func (net *Network) Feedforward(x []float64) []float64 {
	in := x
	for _, L := range net.layers {
		in = L.Forward(in)
	}
	return in
}

func (net *Network) Train(samples [][]float64, targets [][]float64, epochs int, lr float64) {
	if len(samples) != len(targets) {
		panic("Количество образцов и целей должно совпадать")
	}

	for e := 0; e < epochs; e++ {
		totalLoss := 0.0

		for idx, x := range samples {
			y := targets[idx]

			activations := make([][]float64, 0)
			zValues := make([][]float64, 0)
			in := x
			activations = append(activations, in)

			for _, L := range net.layers {
				a := L.Forward(in)
				zValues = append(zValues, L.zs[0])
				activations = append(activations, a)
				in = a
			}

			output := activations[len(activations)-1]

			for i := 0; i < len(y); i++ {
				diff := output[i] - y[i]
				totalLoss += 0.5 * diff * diff
			}

			deltas := make([][]float64, len(net.layers))

			Lout := net.layers[len(net.layers)-1]
			deltaOut := make([]float64, Lout.out)
			for i := 0; i < Lout.out; i++ {
				a := output[i]
				z := zValues[len(zValues)-1][i]
				deltaOut[i] = (a - y[i]) * sigmoidPrime(z)
			}
			deltas[len(net.layers)-1] = deltaOut

			for l := len(net.layers) - 2; l >= 0; l-- {
				L := net.layers[l]
				Lnext := net.layers[l+1]
				delta := make([]float64, L.out)

				for i := 0; i < L.out; i++ {
					sum := 0.0
					for k := 0; k < Lnext.out; k++ {
						sum += Lnext.weights[k][i] * deltas[l+1][k]
					}
					z := zValues[l][i]
					delta[i] = sum * sigmoidPrime(z)
				}

				deltas[l] = delta
			}

			for l := 0; l < len(net.layers); l++ {
				L := net.layers[l]
				aPrev := activations[l]
				for i := 0; i < L.out; i++ {
					L.biases[i] -= lr * deltas[l][i]
					for j := 0; j < L.in; j++ {
						grad := deltas[l][i] * aPrev[j]
						L.weights[i][j] -= lr * grad
					}
				}
			}
		}

		if e%1000 == 0 || e == epochs-1 {
			avgLoss := totalLoss / float64(len(samples))
			fmt.Printf("Эпоха %d: средняя ошибка = %.6f\n", e, avgLoss)
		}
	}
}

func (net *Network) Predict(x []float64) []float64 {
	return net.Feedforward(x)
}
