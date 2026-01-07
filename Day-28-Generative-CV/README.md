# Day 28 - Generative Computer Vision
 
 **Topics Covered:** Autoencoders, Latent Space, Denoising AE, GANs (Generative Adversarial Networks), Generator vs Discriminator, Mode Collapse, StyleGAN
 
 ---
 
 ## Question 1: Discriminative vs Generative Models
 
 **Topic:** Concept
 **Difficulty:** Basic
 
 ### Question
 What is the fundamental difference between Discriminative and Generative models?
 
 ### Answer
 
 - **Discriminative ($P(Y|X)$):** "Classify this image."
    - Given X (Image), predict Y (Label: Cat).
    - Learns the boundary between classes.
 - **Generative ($P(X)$ or $P(X, Y)$):** "Create an image."
    - Learns the probability distribution of the data itself.
    - Can generate new samples from that distribution.
 
 ---
 
 ## Question 2: Autoencoder Architecture
 
 **Topic:** Architecture
 **Difficulty:** Basic
 
 ### Question
 An Autoencoder learns $Output = Input$. Why is this useful?
 
 ### Answer
 
 **Usefulness comes from the Bottleneck.**
 - **Encoder:** Compresses input (784 pixels) to a small latent vector (e.g., 32 dims).
 - **Decoder:** Reconstructs the input from the vector.
 - Since the vector is small, the network is forced to learn **meaningful features** (edges, shapes) to compress the data efficiently.
 - **Uses:** Dimensionality Reduction, Denoising, Anomaly Detection.
 
 ---
 
 ## Question 3: Denoising Autoencoder
 
 **Topic:** Application
 **Difficulty:** Intermediate
 
 ### Question
 How do you train an Autoencoder to remove noise from old photos?
 
 ### Answer
 
 1. **Data:** Take a clean image ($X$).
 2. **Noise:** Add random Gaussian noise to it ($X' = X + \text{noise}$).
 3. **Input:** Feed $X'$ (Noisy) to the Encoder.
 4. **Target:** Force the Decoder to output $X$ (Clean).
 5. **Result:** The model learns that noise is "useless information" that shouldn't be encoded, effectively learning to filter it out.
 
 ---
 
 ## Question 4: GAN (Generative Adversarial Network)
 
 **Topic:** Concept
 **Difficulty:** Intermediate
 
 ### Question
 Explain the "Adversarial Game" between logic of the Generator (G) and Discriminator (D).
 
 ### Answer
 
 **Analogy: Counterfeiter vs Police.**
 - **Generator (G):** Tries to create fake currency (images) to fool the Police.
 - **Discriminator (D):** Tries to distinguish between Real currency (Training Data) and Fake currency (G's output).
 - **Training:**
    1. Train D to maximize accuracy (Real=1, Fake=0).
    2. Train G to minimize D's accuracy (Make D say Real=1 for fakes).
 - **Nash Equilibrium:** Ideally, G becomes perfect, and D guesses 50/50.
 
 ---
 
 ## Question 5: Latent Space
 
 **Topic:** Concept
 **Difficulty:** Advanced
 
 ### Question
 What is "walking in the latent space"?
 
 ### Answer
 
 - The Generator takes a random noise vector $z$ (Latent Vector) as input.
 - This vector represents features (e.g., pose, smile, hair color).
 - **Walking:** Linearly interpolating between two vectors $z_1$ (Man) and $z_2$ (Woman).
 - **Result:** The generated image smoothly morphs from Man -> Androgynous -> Woman, proving the model learned the manifold of human faces.
 
 ---
 
 ## Question 6: Mode Collapse
 
 **Topic:** Challenges
 **Difficulty:** Advanced
 
 ### Question
 What is "Mode Collapse" in GAN training?
 
 ### Answer
 
 **Failure Mode:**
 - The Generator finds *one specific image* that fools the Discriminator successfully (e.g., a specific blurry dog).
 - It generates *only* that image every time (ignoring all other breeds/poses).
 - The Discriminator learns to block it, G moves to another single spot.
 - Loss oscillates, and diversity is lost.
 
 ---
 
 ## Question 7: GAN Loss Function (Minimax)
 
 **Topic:** Math
 **Difficulty:** Advanced
 
 ### Question
 Write the standard Minimax Loss function for GANs.
 
 ### Answer
 
 $$ \min_G \max_D V(D, G) = \mathbb{E}_{x \sim data}[\log D(x)] + \mathbb{E}_{z \sim noise}[\log(1 - D(G(z)))] $$
 
 - **Term 1:** D wants to maximize prob of Real data ($D(x)$ close to 1).
 - **Term 2:** D wants to maximize prob of predicting Fake for generated data ($1 - D(G(z))$ close to 1). G wants to minimize this (fool D).
 
 ---
 
 ## Question 8: CycleGAN
 
 **Topic:** Architecture
 **Difficulty:** Advanced
 
 ### Question
 How does CycleGAN perform Horse -> Zebra translation without Paired Data?
 
 ### Answer
 
 **Cycle Consistency:**
 - We don't have exact "Horse A" = "Zebra A" photos.
 - **Logic:**
    1. $G_{AB}$: Translate Horse -> Zebra.
    2. $G_{BA}$: Translate Zebra -> Horse.
    3. **Constraint:** If I translate Horse -> Zebra -> Horse, I should get the original image back. ($X \approx G_{BA}(G_{AB}(X))$).
 - This preserves smoothness and content.
 
 ---
 
 ## Question 9: Neural Style Transfer
 
 **Topic:** Application
 **Difficulty:** Intermediate
 
 ### Question
 How does Style Transfer (Prisma app) work using VGG?
 
 ### Answer
 
 It is an Optimization problem, not a training problem.
 1. Start with a random noise image.
 2. **Content Loss:** Minimize MSE between Feature Maps of noise and Content Image (preserve shape).
 3. **Style Loss:** Minimize difference in Gram Matrices (texture/correlations) between noise and Style Photo (preserve artistic style).
 4. Gradient Descent updates pixel values of the noise image until it looks right.
 
 ---
 
 ## Question 10: Deepfakes
 
 **Topic:** Ethics
 **Difficulty:** Basic
 
 ### Question
 What is the core technology behind Deepfakes? What is the main ethical concern?
 
 ### Answer
 
 **Tech:** Usually Autoencoders or GANs.
 - Train an AutoEncoder on Person A (Trump).
 - Train an AutoEncoder on Person B (Cage).
 - Swap the Decoders. Feed Person A's face into Person B's decoder.
 
 **Ethics:**
 - Non-consensual pornography.
 - Political disinformation.
 - Fraud/Identity theft.
 
 ---
 
 ## Key Takeaways
 
 - **Autoencoders:** Compress data; good for denoising/anomaly detection.
 - **GANs:** Generate data via competition (Minimax Game).
 - **Latent Space:** The hidden manifold where "meaning" lives.
 - **Training Instability:** GANs are notoriously hard to train (Mode Collapse).
 
 **Next:** [Weeks 5 & 6 Coming Soon](../README.md)
