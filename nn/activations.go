package nn

import (
	"math"
	"time"
)

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func SigmoidPrime(x float64) float64 {
	s := Sigmoid(x)
	return s * (1 - s)
}

func RandomFloat() float64 {
	return randFloat()*2 - 1
}

func randFloat() float64 {
	return float64(mathRAND()) / (1<<63 - 1)
}

// обёртка чтобы не импортировать math/rand здесь
var rng = mathRandInit()

func mathRandInit() func() int64 {
	return func() int64 { return timeNow() }
}

func mathRAND() int64 {
	return rng()
}

func timeNow() int64 {
	return time.Now().UnixNano()
}
