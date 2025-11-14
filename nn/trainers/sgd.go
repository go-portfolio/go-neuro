package trainers

import (
	"fmt"
	"github.com/go-portfolio/go-neuro/nn"
)

type SGDTrainer struct {
	LearningRate float64
}

func (t *SGDTrainer) Update(net *nn.Network, samples, targets [][]float64, epochs int) {
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0

		for idx, x := range samples {
			y := targets[idx]

			activations := [][]float64{x}
			zs := [][]float64{}

			in := x
			for _, L := range net.Layers {
				a := L.Forward(in)
				activations = append(activations, a)
				zs = append(zs, append([]float64{}, L.Z...))
				in = a
			}

			output := activations[len(activations)-1]

			for i := range y {
				diff := output[i] - y[i]
				totalLoss += 0.5 * diff * diff
			}

			deltas := make([][]float64, len(net.Layers))

			last := len(net.Layers) - 1
			deltas[last] = make([]float64, net.Layers[last].Out)
			for i := 0; i < net.Layers[last].Out; i++ {
				a := output[i]
				z := zs[last][i]
				deltas[last][i] = (a - y[i]) * nn.SigmoidPrime(z)
			}

			for l := last - 1; l >= 0; l-- {
				L := net.Layers[l]
				Lnext := net.Layers[l+1]

				delta := make([]float64, L.Out)
				for i := 0; i < L.Out; i++ {
					sum := 0.0
					for k := 0; k < Lnext.Out; k++ {
						sum += Lnext.Weights[k][i] * deltas[l+1][k]
					}
					z := zs[l][i]
					delta[i] = sum * nn.SigmoidPrime(z)
				}
				deltas[l] = delta
			}

			for l := 0; l < len(net.Layers); l++ {
				L := net.Layers[l]
				aPrev := activations[l]

				for i := 0; i < L.Out; i++ {
					L.Biases[i] -= t.LearningRate * deltas[l][i]
					for j := 0; j < L.In; j++ {
						L.Weights[i][j] -= t.LearningRate * deltas[l][i] * aPrev[j]
					}
				}
			}
		}

		if epoch%1000 == 0 || epoch == epochs-1 {
			fmt.Printf("Epoch %d, loss=%.6f\n", epoch, totalLoss/float64(len(samples)))
		}
	}
}
