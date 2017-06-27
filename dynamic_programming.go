package main

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
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
		if rand.NormFloat64()-.5 > 0 {
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
		acc := randNormVector(dim)         // noisy acceleration vector
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

type CostChoicesPair struct {
	choices []int
	cost    float64
}

type Cache struct {
	cost            *mat64.Dense
	choice          *mat64.Dense
	costWithChoices *mat64.Dense
}

func loadCost(cache *Cache, start int, stop int) (float64, bool) {
	cost := cache.cost.At(start, stop)
	return cost, !math.IsNaN(cost)
}

func storeCost(cache *Cache, start int, stop int, cost float64) {
	cache.cost.Set(start, stop, cost)
}

func loadBestChoice(cache *Cache, nChoices int,
	start int) (CostChoicesPair, bool) {
	choice := cache.choice.At(nChoices, start)
	cost := cache.costWithChoices.At(nChoices, start)
	switch {
	case math.IsNaN(choice) && math.IsNaN(cost):
		return CostChoicesPair{}, false // cache miss
	case !math.IsNaN(choice) && !math.IsNaN(cost):

		// walk backward through cache selecting best choices
		choice := int(choice)
		choices := []int{choice}

		for choiceNum := nChoices - 1; choiceNum > 0; choiceNum-- { // (nChoices, 0)
			choice := cache.choice.At(choiceNum, choice)
			choices = append(choices, int(choice))
			if math.IsNaN(choice) {
				panic(fmt.Sprintf("Encountered a NaN choice at index (%d, %d).",
					choiceNum, choice))
			}
		}
		return CostChoicesPair{choices, cost}, true
	default:
		panic("Caches out of sync")
	}
}

func storeBestChoice(cache *Cache, nChoices int, start int,
	value CostChoicesPair) {
	if len(value.choices) > 0 {
		lastChoice := value.choices[0]
		cache.choice.Set(nChoices, start, float64(lastChoice))
	} else {
		panic("Attempted to store an empty list of choices")
	}
	cache.costWithChoices.Set(nChoices, start, value.cost)
}

func getCost(path *mat64.Dense, start int, stop int, cache *Cache) float64 {

	// check that stop and start haven't gotten goofed up
	size, dim := path.Dims()
	if start < 0 || stop > size {
		panic(fmt.Sprintf("Start (%d) must be positive and stop (%d) must be less "+
			"than path length (%d).", start, stop, size))
	}

	// check if return cost has been cached
	cost, ok := loadCost(cache, start, stop)
	if ok {
		return cost
	}

	// accumulate cost of path
	if start + 1 == 0 {
		return 0.0
	}
	cost = 0.0
	for i := start + 1; i < stop; i++ {
		diffsSq := 0.0
		for j := 0; j < dim; j++ {
			diff := path.At(i, j) - path.At(start, j)
			diffsSq += math.Pow(diff, 2)
		}
		cost += math.Sqrt(diffsSq)
	}
	storeCost(cache, start, stop, cost)
	return cost
}

func _bestChoice(nChoices int, path *mat64.Dense, start int, cache *Cache) CostChoicesPair {

	// check that start is a valid number
	stop, _ := path.Dims()
	if start >= stop {
		panic(fmt.Sprintf("Start (%d) must be positive and strictly less than"+
			"path length (%d)", start, stop))
	}

	// check if return value has been cached
	value, ok := loadBestChoice(cache, nChoices, start)
	if ok {
		return value
	}

	// terminal condition: ran out of choices
	if nChoices == 0 {
		return CostChoicesPair{nil, getCost(path, start, stop, cache)}
	}

	// find the choice of i that minimizes cost
	minCost := math.Inf(1)
	var bestChoices []int
	for i := start; i < stop; i++ { // [start, stop)
		costBefore := getCost(path, start, i, cache)
		sliceAfter := _bestChoice(nChoices-1, path, i, cache)
		cost := costBefore + sliceAfter.cost
		if cost < minCost {
			minCost = cost

			// prepend most recent choice
			bestChoices = append([]int{i}, sliceAfter.choices...)
		}
	}

	value = CostChoicesPair{bestChoices, minCost}
	storeBestChoice(cache, nChoices, start, value)
	return value
}

func newCacheMatrix(height int, width int) *mat64.Dense {
	matrix := mat64.NewDense(height, width, nil)
	matrix.Scale(math.NaN(), matrix)
	return matrix
}

//func bestChoicesWithCache()

func bestChoices(nChoices int, path *mat64.Dense) (float64, []int) {
	pathLen, _ := path.Dims()
	cache := Cache{}

	// store the costs for all intervals [i:j] in path.
	cache.cost = newCacheMatrix(pathLen, pathLen+1)

	// Store the most recent choice for choice number i and path interval starting
	// at j.
	cache.choice = newCacheMatrix(nChoices+1, pathLen+1)

	// Store the cost, taking into account i subsequent choices, for path interval
	// starting at j.
	cache.costWithChoices = newCacheMatrix(nChoices+1, pathLen+1)

	value := _bestChoice(nChoices, path, 0, &cache) // actual computation
	fmt.Println("Choices:")
	fmt.Println(mat64.Formatted(cache.choice))
	fmt.Println("Cost with choices:")
	fmt.Println(mat64.Formatted(cache.costWithChoices))
	return value.cost, value.choices
}

// TODO: why are the values not in sorted order?
// TODO: why are values repeated even when cost is not zero?

func main() {
	rand.Seed(2)
	walk := simpleRandomWalk(8)
	fmt.Println(mat64.Formatted(walk.T()))
	cost, choices := bestChoices(5, walk)
	fmt.Println(choices)
	fmt.Println(cost)
}
