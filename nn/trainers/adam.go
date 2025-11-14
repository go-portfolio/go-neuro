package trainers

import "github.com/go-portfolio/go-neuro/nn"

type AdamTrainer struct {
	LearningRate float64
}

func (t *AdamTrainer) Update(net *nn.Network, samples, targets [][]float64, epochs int) {
	// Реализация будет позже
	panic("AdamTrainer not implemented yet")
}
