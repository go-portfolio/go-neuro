package nn

import (
	"math"
	"math/rand"
)

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidPrime(x float64) float64 {
	s := sigmoid(x)
	return s * (1 - s)
}

func randomFloat() float64 {
	return rand.Float64()*2 - 1
}
