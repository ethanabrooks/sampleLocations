package main

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"math"
	"math/rand"
	"sync"
	"runtime"
)

var INF = math.Inf(1)

func randNormVector(size int) *mat64.Vector {
	vector := mat64.NewVector(size, nil)
	for i := 0; i < size; i++ {
		vector.SetVec(i, rand.NormFloat64()*0.001)
	}
	return vector
}

func setRow(i int, m *mat64.Dense, v *mat64.Vector) {
	_, dim := m.Dims()
	for j := 0; j < dim; j++ {
		m.Set(i, j, v.At(j, 0))
	}
}

func simpleRandomWalk(steps int) *mat64.Dense {
	positions := mat64.NewDense(steps, 1, nil)
	vel := 0
	for i := 1; i < steps; i++ {
		acc := rand.Intn(5)
		if rand.NormFloat64() - .5 > 0 {
			vel += acc
		} else {
			vel -= acc
		}
		oldPos := int(positions.At(i-1, 0))
		positions.Set(i, 0, float64(oldPos+vel))
	}
	return positions
}

func randomWalk(steps int, dim int) *mat64.Dense {
	positions := mat64.NewDense(steps, dim, nil)
	newPos := mat64.NewVector(dim, nil)
	vel := mat64.NewVector(dim, nil)
	for i := 1; i < steps; i++ {
		acc := randNormVector(dim)         // noisey acceleration vector
		vel.AddVec(vel, acc)               // vel += acc
		noise := rand.ExpFloat64() / 0.5   // makes path jagged
		vel.ScaleVec(noise, vel)           // vel *= noise
		oldPos := positions.RowView(i - 1) // oldPos = positions[i - 1]
		newPos.AddVec(oldPos, vel)         // newPos = oldPos + vel
		setRow(i, positions, newPos)       // positions[i] = newPos
	}
	return positions
}

func reverse(numbers []int) []int {
	for i := 0; i < len(numbers)/2; i++ {
		j := len(numbers) - i - 1
		numbers[i], numbers[j] = numbers[j], numbers[i]
	}
	return numbers
}

type CacheValue struct {
	choices []int
	cost    float64
}

type Cache struct {
	cost *mat64.Dense
	choice *mat64.Dense
}
var cacheSize uint32 = 0

func load(cache *Cache, i int, j int, includeChoices bool) (CacheValue, bool) {
	cost := cache.cost.At(i, j)
	choice := cache.choice.At(i, j)

	switch {
	case math.IsNaN(cost) && !math.IsNaN(choice):
		panic("Caches out of sync") // every choice should have an associated cost
	case math.IsNaN(cost) || math.IsNaN(choice) && includeChoices:
		return CacheValue{}, false // desired values are infs means cache miss
	case !math.IsNaN(cost) && !includeChoices:   // whether or not math.IsNaN(choice)
		return CacheValue{nil, cost}, true // we only care about cost: ignore missing choice
	default: //case !math.IsNaN(cost) && includeChoices:
		// walk backward through cache selecting best choices
		fmt.Println("Cost:", cost, "includeChoices:", includeChoices)
		var choices []int
		for choiceNum := i - 1; i >= 0; i-- {
			choice = cache.choice.At(choiceNum, int(choice))
			choices = append(choices, int(choice))
		}
		return CacheValue{choices, cost}, true
	}
}

func store(cache *Cache, i int, j int, includesChoices bool, value CacheValue) {
	if includesChoices && len(value.choices) > 0 {
		lastChoice := value.choices[len(value.choices) - 1]
		cache.choice.Set(i, j, float64(lastChoice))
		cache.cost.Set(i, j, value.cost)
	}
	if !includesChoices {
		cache.cost.Set(i, j, value.cost)
	}
}


func getCost(path *mat64.Dense, start int, stop int, cache *Cache) float64 {

	// check if return value has been cached
	value, ok := load(cache, start, stop, false)
	if ok {
		return value.cost
	}

	// check that stop and start haven't gotten goofed up
	size, dim := path.Dims()
	if start < 0 || stop > size {
		panic(fmt.Sprintf("start (%d) must be positive and stop (%d) must be less "+
			"than path length (%d)", start, stop, size))
	}

	// accumulate cost of path
	cost := 0.0
	for i := start + 1; i < stop; i++ {
		diffsSq := 0.0
		for j := 0; j < dim; j++ {
			diff := path.At(i, j) - path.At(start, j)
			diffsSq += math.Pow(diff, 2)
		}
		cost += math.Sqrt(diffsSq)
	}
	store(cache, start, stop, false, CacheValue{nil, cost})
	return cost
}

func _bestChoice(nChoices int, path *mat64.Dense, start int, cache *Cache) CacheValue {

	 //check if return value has been cached
	value, ok := load(cache, nChoices, start, true)
	if ok {
		return value
	}

	stop, _ := path.Dims()

	// if not cached, calculate
	if nChoices == 0 { // running out of choices is terminal condition
		return CacheValue{nil, getCost(path, start, stop, cache)}
	}

	// find the choice of i that minimizes cost
	minCost := math.Inf(1)
	var bestChoices []int

	indices := make(chan int)
	go func() {
		for i := start + 1; i < stop; i++ {
			indices <- i
		}
		close(indices)
	}()

	var group sync.WaitGroup
	for i:= 0; i < runtime.NumCPU(); i++ {
		group.Add(1)
		//go func() {
		func() {
			defer group.Done()
			for j := range indices {
				sliceAfter := _bestChoice(nChoices-1, path, j, cache)
				cost := getCost(path, start, j, cache) + sliceAfter.cost
				if cost < minCost {
					minCost = cost
					bestChoices = append(sliceAfter.choices, j)
				}
			}
		}()
	}
	group.Wait()
	value = CacheValue{bestChoices, minCost}
	store(cache, start, stop, true, value)
	return value
}

func bestChoice(nChoices int, path *mat64.Dense) (float64, []int) {
	pathLen, _ := path.Dims()
	cache := Cache{}
	cache.cost = mat64.NewDense(pathLen, pathLen + 1, nil)
	cache.cost.Scale(INF, cache.cost)
	cache.choice = mat64.NewDense(pathLen, pathLen + 1, nil)
	cache.choice.Scale(INF, cache.cost)
	value := _bestChoice(nChoices, path, 0, &cache)
	fmt.Println(mat64.Formatted(cache.choice))
	fmt.Println(mat64.Formatted(cache.cost))
	return value.cost, reverse(value.choices)
}

func main() {
	rand.Seed(0)
	walk := simpleRandomWalk(7)
	choices, cost := bestChoice(3, walk)
	fmt.Println(mat64.Formatted(walk.T()))
	fmt.Println(choices)
	fmt.Println(cost)
	fmt.Println(cacheSize)
}
