package trainers

import "github.com/go-portfolio/go-neuro/nn"

type Trainer interface {
    Train(net *nn.Network, samples [][]float64, targets [][]float64, epochs int)
}
