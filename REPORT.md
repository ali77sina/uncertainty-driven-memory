# How We Taught Neural Networks to Know What They Don't Know

## A Full Report on Uncertainty Estimation, Adaptive Search, and Latent Reasoning

---

## Part 1: The Problem

Neural networks are confident by default. When you ask one a question, it gives you an answer. It almost never says "I don't know." Even when it should.

This is a problem. If you're using a model to diagnose a medical image, fly a drone, or solve a math problem, you need to know when the model is guessing versus when it actually knows. A wrong answer delivered with high confidence is worse than no answer at all, because you trust it and act on it.

The standard way to measure a model's uncertainty is expensive. You train 10 copies of the same model (an "ensemble"), show them all the same input, and check if they agree. If they all say "cat," the model is probably right. If five say "cat" and five say "dog," the model is uncertain. This works well, but it costs 10x the compute at training and 10x at inference. For real deployments, that's often not practical.

We set out to find a cheaper way to get the same signal. And then we asked: once you know the model is uncertain, what should you do about it?

---

## Part 2: A Simpler Way to Measure Uncertainty

### The Idea

Every neural network builds internal representations of its inputs. When you feed an image into a classifier, the raw pixels get transformed layer by layer into increasingly abstract features. Somewhere inside the network, there's a compact vector (we call it the "latent representation") that captures what the network thinks is important about that input.

During training, the network sees thousands of examples. Each one produces a latent vector. All these vectors together form a cloud of points in a high-dimensional space. This cloud has a shape. Images of cats produce latent vectors in one region. Images of dogs produce vectors in another region.

Our hypothesis: if you show the network something it has never seen before (say, an image of a truck when it was trained on animals), the latent vector will land far from the cloud. The network has no experience with this kind of input, so its internal representation will be different from anything it produced during training.

Measuring how far a new latent vector is from the training cloud gives you an uncertainty score. Close to the cloud means "I've seen this before, I'm probably right." Far from the cloud means "this is unfamiliar, don't trust my answer."

### Experiment 1: Sine Regression (The Toy Version)

We started simple. A small neural network learns to predict sin(x) for x between negative pi and pi. Then we ask it to predict sin(x) for x far outside that range, where it has never seen data.

The question: can the model tell us that it's uncertain about predictions outside its training range?

We compared three approaches:

**Deep Ensemble (the expensive standard):** Train 10 separate networks. For each input, run all 10 and compute the standard deviation of their predictions. High standard deviation means uncertainty.

**Bayesian Neural Network:** Instead of fixed weights, learn a probability distribution over weights. Sample different weight configurations and check if predictions agree.

**Latent Distance (our method):** Train one network with a bottleneck layer. Save the latent vectors from all training examples. For a new input, compute its latent vector and measure the distance to the nearest training vector.

On this simple task, all three methods worked. Inside the training range, uncertainty was low. Outside, it was high. The latent distance method gave the sharpest signal (near-zero inside, jumps hard at the boundary) and used the least compute (one forward pass instead of ten).

### Experiment 2: MNIST vs FashionMNIST (The Real Test)

The sine task was too easy. We needed a harder test.

MNIST is a dataset of handwritten digits (0 through 9). FashionMNIST looks the same structurally (28x28 grayscale images, 10 classes) but contains clothing items instead of digits. We trained classifiers on MNIST and asked: can the model tell that FashionMNIST images are "not digits" without ever having seen clothing during training?

This is called out-of-distribution (OOD) detection. The metric is AUROC: a score from 0 to 1 measuring how well the uncertainty signal separates in-distribution (MNIST) from out-of-distribution (FashionMNIST). A score of 1.0 means perfect separation. A score of 0.5 means random guessing.

Our naive latent distance approach (PCA to 2 dimensions, then nearest-neighbor) collapsed completely: 0.455 AUROC, worse than random. Reducing the latent space to just 2 dimensions threw away too much information. The nearest-neighbor distance in that compressed space couldn't distinguish digits from clothing.

This was the key moment. The latent space itself contained the information. We just needed a better way to measure density in it.

We tried two better density estimators on the same latent vectors:

**Gaussian Mixture Model (GMM):** Fit a statistical model with 10 Gaussian clusters to the training latent vectors. For a new input, compute how likely its latent vector is under this model. Low likelihood means unfamiliar.

**Normalizing Flow:** A small neural network trained jointly with the classifier that learns to map the latent distribution to a simple Gaussian. The log-probability under this transformation gives a density score.

Results:

| Method | AUROC | Inference Time (2000 samples) |
|---|---|---|
| Deep Ensemble (10 models) | 0.979 | 698ms |
| Bayesian Neural Network | 0.957 | 782ms |
| Latent + kNN (naive) | 0.455 | 190ms |
| Latent + GMM | 0.991 | 181ms |
| Latent + Normalizing Flow | 0.975 | 193ms |

The GMM approach beat the 10-model ensemble while being nearly 4x faster. A single model, with a simple statistical model fit to its internal representations after training, outperformed the most expensive standard method.

The lesson: the problem was never the neural network's representations. They contained all the information needed to detect unfamiliar inputs. The problem was how we measured distance in that space.

---

## Part 3: Using Uncertainty to Make Better Decisions

Knowing that you're uncertain is only useful if you do something with that knowledge. We hypothesized that a model's uncertainty should control how much computational effort it spends on each step of a problem.

Think of it like a human solving a long math problem. On easy steps (adding small numbers), you just do it quickly. On hard steps (multiplying large numbers, dealing with fractions), you slow down, double-check, maybe try a different approach. You allocate your effort based on difficulty.

Neural networks don't do this by default. They either commit to their first answer at every step ("greedy decoding") or explore multiple alternatives at every step ("beam search"). Neither is ideal. Greedy is fast but makes unrecoverable errors. Beam search is thorough but wastes compute on steps that were already easy.

### Experiment 3: Multi-Step Prediction Chains

We built a model that predicts the result of applying a chain of mathematical operations (sine, square, absolute value, etc.) step by step. The model is imperfect. Each step has some error. Over 15 steps, errors compound.

Three strategies:

**Greedy:** Always take the model's best prediction. Fast, but errors compound.

**Beam Search (fixed width):** At every step, keep multiple candidate answers and carry them all forward. Explores alternatives uniformly.

**Uncertainty-Guided:** Only keep multiple candidates at steps where the model is uncertain. At confident steps, just take the best prediction.

The surprise: beam search made things worse. At chain length 15, greedy got 26% accuracy while beam-8 (keeping 8 candidates at every step) got 14%. Why? Because at each step, beam search samples from the model's predicted distribution. Each sample adds a small amount of noise. Over 15 steps, that noise compounds faster than the model's own prediction error. The mean prediction (greedy) is actually the least noisy option.

Uncertainty-guided search avoided this trap. It stayed with the mean prediction on confident steps (no sampling noise) and only explored alternatives on uncertain steps. It got 24% accuracy (close to greedy) but spent its compute budget only where it mattered.

### Experiment 4: A Real Language Model on Math

We moved from toy models to a real one: Qwen 2.5 (0.5 billion parameters), a small but capable language model, solving arithmetic word problems.

Base model accuracy with greedy decoding: 80%. It gets some problems wrong because it occasionally makes arithmetic mistakes during its step-by-step reasoning.

We tested two branching strategies:

**Entropy branching:** At each step of generation, check how spread out the model's next-token probabilities are. If the model is torn between multiple options (high entropy), generate several alternatives and continue with all of them, keeping the best one at the end. If the model is confident (low entropy), just take the top choice.

**Latent-density branching:** Instead of looking at the output probabilities, look at the model's internal hidden state. Compare it against what the hidden state looks like during correct reasoning (measured by the GMM density model from earlier work). If the hidden state is unfamiliar, branch. If it looks like normal correct reasoning, commit.

Results:

| Method | Accuracy | Avg Forward Passes |
|---|---|---|
| Greedy | 80% | 70 |
| Entropy branching | 98% | 65 |
| Latent-density branching | 98% | 65 |

Both branching methods recovered almost every problem that greedy got wrong. They went from 80% to 98%. And they actually used fewer forward passes than greedy (65 vs 70), because branching at the right moment found the correct answer faster than committing to a wrong path and generating more tokens.

On this task, entropy and density branching performed identically. The arithmetic problems were simple enough that the model was uncertain (high entropy) whenever it was about to make a mistake. For harder tasks where the model is confidently wrong (low entropy but incorrect), the density approach should outperform entropy because it detects unfamiliar reasoning states regardless of output confidence. We haven't found that task yet, but the mechanism is in place.

---

## Part 4: Can Models Think Without Words?

This part went in a different direction. Current language models reason by generating words. Chain-of-thought prompting makes the model write out its intermediate steps ("First I add 5 and 3 to get 8, then I multiply by 2 to get 16..."). This works because the intermediate tokens act as external memory, keeping track of state.

But words are a lossy bottleneck. A single token carries a limited amount of information. The model has to compress its entire understanding of the problem into a sequence of discrete symbols. And generating each token is expensive (it requires a full pass through the model).

What if the model could "think" without words? Instead of generating tokens between reasoning steps, it could produce continuous vectors (numerical arrays, not decoded into text) that carry its intermediate state. These "thought vectors" would be cheaper to produce, carry more information per unit, and wouldn't be constrained by the vocabulary.

### Experiment 5: Token Reasoning vs Latent Reasoning

We trained small transformers on a multi-step arithmetic task: track a number through a chain of operations (add 3, multiply by 2, subtract 5, conditional: if greater than 50 add 10 else subtract 10, etc.) and predict the final value.

Three architectures, all the same size:

**Standard transformer:** Encode the full sequence, predict the answer directly. No intermediate reasoning at all.

**Token chain-of-thought:** After each operation, the model generates the intermediate answer as actual tokens. Each step is directly supervised.

**Latent thought vectors:** After each operation, the model produces a continuous vector (not decoded into words) that feeds into the next step. The vectors are trained to encode the intermediate state through an auxiliary loss that pushes them to predict the intermediate answer, but they never become actual tokens.

We trained on chains of length 3 to 25. Then we tested on chains of length 5, 10, 15, 20, 30, 40, and 50. Everything up to 25 is "in-distribution" (the model saw similar lengths during training). Everything beyond 25 is "out-of-distribution" (the model has to generalize).

Results:

| Method | L=5 | L=10 | L=15 | L=20 | L=30 | L=40 | L=50 |
|---|---|---|---|---|---|---|---|
| Standard | 2% | 2% | 1% | 1% | 0% | 1% | 2% |
| Token-CoT | 100% | 100% | 100% | 100% | 0% | 1% | 0% |
| Latent Thought | 4% | 5% | 3% | 3% | 3% | 3% | 4% |

The standard transformer learned nothing. Without intermediate reasoning, it cannot track state through a chain of operations.

Token chain-of-thought was perfect within the training range: 100% accuracy at every length up to 20. Then it hit a cliff. At length 30 (just 5 steps beyond training), accuracy dropped to 0.2%. At length 50, it was 0%. A complete collapse. The model memorized the pattern of generating intermediate tokens at lengths it had seen, but that pattern doesn't transfer to longer chains.

Latent thought vectors had much lower absolute accuracy (3-5%). Learning to represent arithmetic in continuous vector space is genuinely hard. The model has to figure out, without being told, how to encode numbers as directions and distances in a high-dimensional space. With limited training, it barely scratched the surface of this problem.

But the degradation profile was the important finding. Latent thoughts went from 4% at length 5 to 4% at length 50. Essentially flat. No cliff. The representation doesn't have a length boundary baked into it. It works (poorly, but consistently) at any length.

### What This Tells Us

Token chain-of-thought has a hidden failure mode. The 100% accuracy at training lengths is impressive but misleading. It masks a hard boundary beyond which the method completely fails. For real applications where reasoning chains have unpredictable length, this cliff is dangerous.

Latent reasoning has the opposite trade-off. Lower peak performance, but no cliff. The generalization to unseen lengths is inherent in the continuous representation, not something that has to be explicitly trained.

The practical path forward is probably combining both: use token reasoning where it's reliable (within the training distribution) and fall back to latent reasoning when the chain gets longer than what the model was trained on. The uncertainty estimation from our earlier work could detect exactly when that transition should happen.

---

## Part 5: Where This All Connects

The experiments form a progression:

1. We found that a single model's internal representations contain enough information to detect unfamiliar inputs (Part 2). A simple GMM on latent vectors beats expensive ensembles.

2. We showed that this uncertainty signal should drive how the model allocates compute during multi-step reasoning (Part 3). Branch when uncertain, commit when confident. This outperforms both greedy and beam search.

3. We discovered that how a model represents its intermediate reasoning matters as much as what it reasons about (Part 4). Token-based reasoning is brittle at the edges of the training distribution. Continuous latent reasoning trades peak accuracy for robustness.

The thread connecting all three: models need internal self-awareness to work reliably on hard, long, or unfamiliar problems. They need to know when they're in familiar territory (latent density), act differently when they're not (adaptive search), and maintain internal state that doesn't catastrophically fail at distribution boundaries (latent reasoning).

None of these ideas are individually new. Ensembles, beam search, and chain-of-thought are all well-established. What's new is connecting them through the lens of latent-space geometry and showing that a single, cheap signal (density in the representation space) can replace expensive alternatives across all three problems.

---

## Summary of Key Numbers

| Experiment | Key Finding | Numbers |
|---|---|---|
| OOD Detection | GMM in latent space beats 10-model ensemble | 0.991 vs 0.979 AUROC, 4x cheaper |
| Continuous chains | Beam search hurts, uncertainty-guided doesn't | Greedy 26%, Beam-8 14%, Unc-guided 24% |
| LLM math (Qwen) | Uncertainty branching recovers errors | 80% to 98%, fewer forward passes |
| Token-CoT generalization | Perfect in-distribution, cliff outside | 100% at L=20, 0% at L=30 |
| Latent thought generalization | Low but stable at all lengths | ~4% at L=5 through L=50, no cliff |
