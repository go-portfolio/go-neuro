package dataset

import "fmt"

type DataSet struct {
	Samples [][]float64
	Targets [][]float64
}

var registry = map[string]DataSet{}

func Register(name string, ds DataSet) {
	registry[name] = ds
}

func MustGet(name string) DataSet {
	ds, ok := registry[name]
	if !ok {
		panic(fmt.Sprintf("dataset '%s' not found", name))
	}
	return ds
}
