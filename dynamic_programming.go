package main

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"math"
	"math/rand"
)

// Generates a random vector size `int`.
func randNormVector(size int) *mat64.Vector {
	vector := mat64.NewVector(size, nil)
	for i := 0; i < size; i++ {
		vector.SetVec(i, rand.NormFloat64()*0.001)
	}
	return vector
}

// Sets row `i` of `m` to the values in `v`.
func setRow(i int, m *mat64.Dense, v *mat64.Vector) {
	dimV, _ := v.Dims()
	sizeM, dimM := m.Dims()
	if dimV != dimM {
		panic(fmt.Sprintf("`v` is size %d. It must be size %d because `m` is %d x " +
			"%d.", dimV, dimM, sizeM, dimM))
	}
	for j := 0; j < dimM; j++ {
		m.Set(i, j, v.At(j, 0))
	}
}

// Generates a random path length `steps` containing all whole number values.
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

// Generates a random path with dimensions `steps` x `dim`.
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

// Reverses an array of ints. Copied from the go cookbook:
// http://golangcookbook.com/chapters/arrays/reverse/
func reverse(numbers []int) []int {
	for i := 0; i < len(numbers)/2; i++ {
		j := len(numbers) - i - 1
		numbers[i], numbers[j] = numbers[j], numbers[i]
	}
	return numbers
}

type cache struct {
	cost            *mat64.Dense
	choice          *mat64.Dense
	costWithChoices *mat64.Dense
}

func loadCost(cache *cache, start int, stop int) (float64, bool) {
	cost := cache.cost.At(start, stop)
	return cost, !math.IsNaN(cost)
}

func storeCost(cache *cache, start int, stop int, cost float64) {
	cache.cost.Set(start, stop, cost)
}

func loadBestChoice(cache *cache, nChoices int,
	start int) (int, float64, bool) {
	choice := cache.choice.At(nChoices, start)
	cost := cache.costWithChoices.At(nChoices, start)
	switch {
	case math.IsNaN(choice) && math.IsNaN(cost):
		return 0, 0, false // cache miss
	case !math.IsNaN(choice) && !math.IsNaN(cost):
		return int(choice), cost, true
	default:
		panic("Caches out of sync")
	}
}

func storeBestChoice(cache *cache, nChoices int, start int, choice int, cost float64) {
	cache.choice.Set(nChoices, start, float64(choice))
	cache.costWithChoices.Set(nChoices, start, cost)
}

func getCost(path *mat64.Dense, start int, stop int, cache *cache) float64 {

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
	if stop - start <= 1 {
		return 0.0
	} else {
		sqDist := 0.0
		for i := 0; i < dim; i++ {
			diff := path.At(start + 1, i) - path.At(start, i)
			sqDist += math.Pow(diff, 2)
		}
		cost := math.Sqrt(sqDist) + getCost(path, start+1, stop, cache)
		storeCost(cache, start, stop, cost)
		return cost

	}
}


func newCacheMatrix(height int, width int) *mat64.Dense {
	matrix := mat64.NewDense(height, width, nil)
	matrix.Scale(math.NaN(), matrix)
	return matrix
}

func nextChoice(nChoices int, path *mat64.Dense, start int,
	cache *cache) (int, float64) {
	stop, _ := path.Dims()

	// check validity of args
	if start < 0 || start >= stop {
		panic(fmt.Sprintf("Start (%d) must be positive and less than"+
			"path length (%d)", start, stop))
	}
	if nChoices <= 0 {
		panic("`nChoices` must be greater than 0")
	}

	// check if return value has been cached
	choice, cost, ok := loadBestChoice(cache, nChoices, start)
	if ok {
		return choice, cost
	}

	// find the choice of i that minimizes cost
	minCost := math.Inf(1)
	var bestChoice int
	for i := start; i < stop; i++ { // [start, stop)
		costBefore := getCost(path, start, i, cache)

		// costAfter value depends on whether there are choices remaining
		var costAfter float64
		if nChoices == 1 {
			costAfter = getCost(path, i, stop, cache)
		} else {
			_, costAfter = nextChoice(nChoices - 1, path, i, cache)
		}

		cost := costBefore + costAfter
		if cost < minCost {
			minCost = cost
			bestChoice = i
		}
	}

	storeBestChoice(cache, nChoices, start, bestChoice, minCost)
	return bestChoice, minCost

}

func bestChoicesWithCache(nChoices int, path *mat64.Dense, start int,
	cache *cache) ([]int, float64) {
	stop, _ := path.Dims()
	if nChoices == 0 {
		return []int{}, getCost(path, start, stop, cache)
	}
	nextChoice, cost := nextChoice(nChoices, path, start, cache)
	subsChoices, _ := bestChoicesWithCache(nChoices  - 1, path, nextChoice, cache)
	return append(subsChoices, nextChoice), cost
}

func bestChoices(nChoices int, path *mat64.Dense) ([]int, float64) {
	pathLen, _ := path.Dims()
	cache := cache{}

	// store the costs for all intervals [i:j] in path.
	cache.cost = newCacheMatrix(pathLen, pathLen+1)

	// Store the most recent choice for choice number i and path interval starting
	// at j.
	cache.choice = newCacheMatrix(nChoices+1, pathLen+1)

	// Store the cost, taking into account i subsequent choices, for path interval
	// starting at j.
	cache.costWithChoices = newCacheMatrix(nChoices+1, pathLen+1)

	// actual computation
	choices, cost := bestChoicesWithCache(nChoices, path, 0, &cache)
	return reverse(choices), cost
}

func main() {
	rand.Seed(11)
	walk := simpleRandomWalk(5)
	fmt.Println(mat64.Formatted(walk.T()))
	cost, choices := bestChoices(3, walk)
	fmt.Println(choices)
	fmt.Println(cost)
}
