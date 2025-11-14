package trainers

import (
    "fmt"
    "github.com/go-portfolio/go-neuro/nn"
)

type SGDTrainer struct {
    LearningRate float64
}

func (t *SGDTrainer) Train(net *nn.Network, samples, targets [][]float64, epochs int) {
    lr := t.LearningRate

    for e := 0; e < epochs; e++ {
        totalLoss := 0.0

        for i := range samples {
            output, activations, zvals := net.ForwardFull(samples[i])
            y := targets[i]

            // loss
            for j := range y {
                d := output[j] - y[j]
                totalLoss += 0.5 * d * d
            }

            deltas := net.Backpropagate(y, activations, zvals)
            net.ApplyGradients(lr, deltas, activations)
        }

        if e%1000 == 0 || e == epochs-1 {
            fmt.Println("Epoch", e, "loss=", totalLoss/float64(len(samples)))
        }
    }
}
