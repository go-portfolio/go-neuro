package dataset

type Dataset struct {
    Name     string
    TaskType string     // "logic", "classification", "regression"
    Samples  [][]float64
    Targets  [][]float64
}

var registry = map[string]Dataset{}

func Register(ds Dataset) {
    registry[ds.Name] = ds
}

func MustGet(name string) Dataset {
    ds, ok := registry[name]
    if !ok {
        panic("dataset not found: " + name)
    }
    return ds
}
