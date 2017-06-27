package main

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"math"
	"math/rand"
)

// Calculates the sum of mean squared errors from the first point in the path.
func getCost(path *mat64.Dense, start int, stop int, cache *cache) float64 {

	// Check validity of parameters.
	size, dim := path.Dims()
	if start < 0 || stop > size {
		panic(fmt.Sprintf("Start (%d) must be positive and stop (%d) must be less "+
			"than path length (%d).", start, stop, size))
	}

	// Check if return cost has been cached.
	cost, ok := loadCost(cache, start, stop)
	if ok {
		return cost
	}

	// Terminal condition: stop is next to start
	if stop-start <= 1 {
		return 0.0
	}

	// Calculate euclidean distance between row at `start` and row at `start+1`.
	sqDistance := 0.0
	for i := 0; i < dim; i++ {
		sqDistance += math.Pow(path.At(start+1, i)-path.At(start, i), 2)
	}
	euclideanDistance := math.Sqrt(sqDistance)
	cost = euclideanDistance + getCost(path, start+1, stop, cache)
	storeCost(cache, start, stop, cost)
	return cost
}

// Returns the tuple `(choice, cost)` where choice is the best next choice for
// `path[start:]` given `nChoices` remaining, and `cost` is the cost for this
// choice.
func nextChoice(
	nChoices int, path *mat64.Dense, start int, cache *cache) (int, float64) {
	stop, _ := path.Dims()

	// Check validity of args.
	if start < 0 || start >= stop {
		panic(fmt.Sprintf("Start (%d) must be positive and less than"+
			"path length (%d)", start, stop))
	}
	if nChoices <= 0 {
		panic("`nChoices` must be greater than 0")
	}

	// Check if return value has been cached.
	choice, cost, ok := loadBestChoice(cache, nChoices, start)
	if ok {
		return choice, cost
	}

	// Find the choice of `i` that minimizes cost.
	minCost := math.Inf(1)
	var bestChoice int
	for i := start; i < stop; i++ { // [start, stop)
		costBefore := getCost(path, start, i, cache)

		// `costAfter` value depends on whether there are choices remaining.
		var costAfter float64
		if nChoices == 1 {
			costAfter = getCost(path, i, stop, cache)
		} else {
			_, costAfter = nextChoice(nChoices-1, path, i, cache)
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

// Returns the tuple `(choices, cost)` where choice is the list of choices
// (length `nChoices`) that minimizes the cost of `path`. Values from
// `nextChoice` and `getCost` are cached in `cache`.
func bestChoicesWithCache(
	nChoices int, path *mat64.Dense, start int, cache *cache) ([]int, float64) {
	stop, _ := path.Dims()
	if nChoices == 0 {
		return []int{}, getCost(path, start, stop, cache)
	}

	// Next choice.
	choice, cost := nextChoice(nChoices, path, start, cache)

	// Subsequent choices.
	otherChoices, _ := bestChoicesWithCache(nChoices-1, path, choice, cache)

	return append(otherChoices, choice), cost
}

// Reverses an array of ints. Copied from the go cookbook:
// http://golangcookbook.com/chapters/arrays/reverse/.
func reverse(numbers []int) []int {
	for i := 0; i < len(numbers)/2; i++ {
		j := len(numbers) - i - 1
		numbers[i], numbers[j] = numbers[j], numbers[i]
	}
	return numbers
}

// Returns the tuple `(choices, cost)` where choice is the list of choices
// (length `nChoices`) that minimizes the cost of `path`.
func bestChoices(nChoices int, path *mat64.Dense) ([]int, float64) {
	pathLen, _ := path.Dims()
	cache := newCache(nChoices, pathLen)
	choices, cost := bestChoicesWithCache(nChoices, path, 0, &cache)
	return reverse(choices), cost
}

func main() {
	rand.Seed(11)
	walk := simpleRandomWalk(500)
	fmt.Println(mat64.Formatted(walk.T()))
	cost, choices := bestChoices(30, walk)
	fmt.Println(choices)
	fmt.Println(cost)
}
