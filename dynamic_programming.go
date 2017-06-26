package main

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"golang.org/x/sync/syncmap"
	"math"
	"math/rand"
)

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
		if rand.NormFloat64() > 0 {
			vel += acc
		} else {
			vel -= acc
		}
		//vel *= noise           // vel *= noise
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

func hashCode(start int, stop int, nChoices ...int) string {
	return fmt.Sprint(start, stop, nChoices)
}

type CacheValue struct {
	choices []int
	cost    float64
}

type Cache syncmap.Map

func getCost(path *mat64.Dense, start int, stop int, cache *syncmap.Map) float64 {
	key := hashCode(start, stop)

	// check if return value has been cached
	value, ok := cache.Load(key)
	if ok {
		return value.(CacheValue).cost
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
	cache.Store(key, CacheValue{nil, cost}) // cache result
	return cost
}

func _bestChoice(nChoices int, path *mat64.Dense, start int, stop int,
	cache *syncmap.Map) CacheValue {

	key := hashCode(start, stop, nChoices)

	// check if return value has been cached
	value, ok := cache.Load(key)
	if ok {
		return value.(CacheValue)
	}

	// if not cached, calculate
	if nChoices == 0 { // running out of choices is terminal condition
		return CacheValue{nil, getCost(path, start, stop, cache)}
	}

	// find the choice of i that minimizes cost
	minCost := math.Inf(1)
	var bestChoices []int

	sem := make(chan bool, stop-start)
	for i := start + 1; i < stop; i++ {
		go func(i int) {
			sliceAfter := _bestChoice(nChoices-1, path, i, stop, cache)
			cost := getCost(path, start, i, cache) + sliceAfter.cost
			if cost < minCost {
				minCost = cost
				bestChoices = append(sliceAfter.choices, i)
			}
			sem <- true
		}(i)
	}
	for i := start + 1; i < stop; i++ {
		<-sem
	}
	value = CacheValue{bestChoices, minCost}
	cache.Store(key, value) // add return value to cache
	return value.(CacheValue)
}

func bestChoice(nChoices int, path *mat64.Dense) ([]int, float64) {
	size, _ := path.Dims()
	value := _bestChoice(nChoices, path, 0, size, &syncmap.Map{})
	return value.choices, value.cost
}

func main() {
	rand.Seed(0)
	walk := simpleRandomWalk(8)
	choices, cost := bestChoice(2, walk)
	fmt.Println(mat64.Formatted(walk.T()))
	fmt.Println(choices)
	fmt.Println(cost)
}
