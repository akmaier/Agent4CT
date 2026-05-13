# Haneda et al. — AAPM CT Metal Artifact Reduction (CT-MAR) Grand Challenge (Med. Phys. 2025)

_Source: <https://pmc.ncbi.nlm.nih.gov/articles/PMC12757780/>_

_PDF: `papers/haneda_2025_ctmar.pdf`_

---

**HHS Public Access** Author manuscript Med Phys. Author manuscript; available in PMC 2026 January 02. 

Published in final edited form as: Med Phys. 2025 October ; 52(10): e70050. doi:10.1002/mp.70050. 

## **AAPM CT metal artifact reduction grand challenge** 

**Eri Haneda**[1] , **Nils Peters**[2,3] , **Jiayong Zhang**[1] , **Grigorios Karageorgos**[1] , **Wenjun Xia**[4] , **Harald Paganetti**[2] , **Ge Wang**[4] , **Yi Guo**[5] , **Jianhua Ma**[5,6] , **Hyoung Suk Park**[7] , **Kiwan Jeon**[7] , **Fuxin Fan**[8] , **Mareike Thies**[8] , **Bruno De Man**[1] 

1GE HealthCare Technology and Innovation Center, Niskayuna, New York, USA 

2Department of Radiation Oncology, Massachusetts General Hospital and Harvard Medical School, Boston, Massachusetts, USA 

3Department of Radiation Oncology, University of Washington & Fred Hutch Cancer Center, Seattle, Washington, USA 

4Department of Biomedical Engineering, Rensselaer Polytechnic Institute, Troy, New York, USA 

5School of Biomedical Engineering, Southern Medical University, Guangdong, China 

6School of Life Science and Technology, Xi’an Jiaotong University, Shaanxi, China 

7Department of Industrial Mathematics, National Institute for Mathematical Sciences, Daejeon, South Korea 

8Pattern Recognition Lab, Friedrich-Alexander-Universität Erlangen-Nürnberg, Erlangen, Germany 

## **Abstract** 

**Background:** Metal artifact reduction (MAR) is a long-standing challenge in CT imaging. The presence of highly attenuating objects, such as dental fillings, hip prostheses, spinal screws/rods, and gold fiducial markers, can introduce severe streak artifacts in CT images, often reducing their diagnostic value. Existing CT MAR studies typically define their own test cases and evaluation metrics, making it difficult to objectively and comprehensively compare the performance of different MAR methods. There is a widespread need for a universal CT MAR image quality benchmark to evaluate the clinical impact of new MAR methods and compare them to state-of-theart techniques. 

**Purpose:** The goal of the AAPM CT Metal Artifact Reduction (CT-MAR) grand challenge was to create and distribute a clinically representative 2D MAR performance benchmark, and to invite participants to objectively compare the performance of their MAR methods based on this benchmark. A secondary goal was to facilitate MAR development by disseminating a MAR training database and tools. After completion of the grand challenge, the MAR benchmark and the MAR training database will remain publicly accessible for future MAR developments and benchmarking. 

> **Correspondence** : Eri Haneda, 1 Research Circle, Niskayuna, NY 12309, USA. haneda@gehealthcare.com. CONFLICTS OF INTEREST STATEMENT The authors have no conflicts to disclose. 

Haneda et al. 

Page 2 

**Methods:** Grand challenge participants were invited to submit results for their MAR algorithm. The challenge organizers provided 14,000 CT training datasets generated using a hybrid data simulation framework that combined real patient images—including lung, abdomen, liver, head, and pelvis—with virtual metal objects. Each training dataset included five types of data: CT sinograms (uncorrected and metal-free), CT reconstructed images (uncorrected and metal-free), and metal masks. In the final evaluation phase, 29 clinical uncorrected datasets with metal were provided in both the sinogram and image domains for participants to process with their MAR algorithms. Their results were evaluated using eight clinically relevant image quality metrics. The final ranking was determined and compared to an established normalized metal artifact reduction (NMAR) reference method. Additionally, we conducted a survey to better understand the methodologies used by participants. 

**Results:** A total of 106 teams registered for the challenge, with 26 teams completing all phases of the challenge. 92% of these—including all top ten teams—used a deep learning (DL) approach, employing a variety of network architectures such as UNet, ResNet, GAN, diffusion models, and transformers. Additionally, 22% of the teams—including the top three teams—utilized a combination of sinogram- and image-domain approaches. The results showed a broad distribution of the scores. Overall, the competition was marked by diverse methods and a wide range of results, including some truly exceptional results. More than 70% of the teams achieved a better overall score than the popular baseline NMAR method. 

**Conclusions:** The CT-MAR grand challenge provided an opportunity to benchmark state-of-theart MAR algorithms. Our hybrid data generation framework was a powerful tool for simulating large-scale realistic datasets for MAR algorithm development. A clinically relevant universal MAR benchmark offered an objective and meaningful way to compare different approaches. The training data and benchmark were published online to support future MAR development. 

## **Keywords** 

AAPM grand challenge; computed tomography; deep learning; metal artifact reduction 

## **1 | INTRODUCTION** 

Metal artifact reduction (MAR) is one of the longest-standing challenges in CT imaging.[1 ] In the presence of highly attenuating objects such as dental fillings, hip prostheses, spinal screws or rods, and gold fiducial marker, CT images are often degraded by streak artifacts. These artifacts can significantly reduce the diagnostic value of the images and also impede their use for intervention or radiation therapy planning, where precise tumor localization and accurate characterization of surrounding tissues are essential.[1,2] Effective MAR significantly enhances clinical usability in several ways. Clearer visualization of anatomy around dental work, orthopedic implants, or surgical clips is important for radiologists to accurately identify and diagnose conditions. Tumors or fractures that might be obscured by metal artifacts could become visible, allowing for precise treatment planning. This is crucial for surgical interventions and radiation therapy, where exact targeting is necessary. Effective metal artifact reduction can eliminate unnecessary repeat scans, avoiding additional patient exposure to radiation. For patients with metal implants, such as joint replacements, MAR 

Med Phys. Author manuscript; available in PMC 2026 January 02. 

Haneda et al. 

Page 3 

allows for better monitoring of treatment progress and implant integrity. This helps in early detection of complications, such as implant loosening or infection.[3] 

The root causes of CT metal artifacts include beam hardening, scattered radiation, noise, photon starvation, and non-linear partial volume (NLPV) effect.[1,4] There are many MAR techniques that address metal artifacts based on various principles. The algorithmic techniques can be classified into five groups: physics-based pre-processing,[5] projection completion,[6–10] statistical model-based reconstruction,[11–15] image post-processing,[16–21 ] and dual-domain processing.[22–28] Physics-based pre-processing corrects projection data by modeling artifact causes such as noise, scatter, beam hardening, and NLPV to more accurately represent the ideal attenuation line integrals. If the data in the metal trace are completely corrupted, projection completion can be performed, which consists of interpolating or inpainting the missing sinogram data in the metal trace. Statistical modelbased reconstruction can be used to either down-weight or fully exclude the corrupted data. It starts with an initial image estimate and iteratively updates it to minimize the mismatch in the projection domain. MAR can also be implemented through post-processing to eliminate artifacts and streaks directly in the reconstructed images. Often, combinations of the above techniques are used to further suppress artifacts. During data acquisition, scan parameters such as X-ray tube voltage, tube current, and scan orientation can also be adjusted to reduce metal artifacts.[29–31] Dualenergy CT, Mega-Voltage CT, and photon-counting CT have been considered for MAR, primarily to eliminate beam hardening and to manage photon-starvation.[32,33] 

Due to recent advances in AI, deep learning (DL) based approaches have also become popular for MAR in CT imaging. While traditional, non-AI approaches need explicit programming and manual design, DL methods involve training models to learn features and patterns of metal artifact related degradation, and automatically generate appropriate corrections of CT projection and/or image data. Although large-scale training datasets are necessary for DL, many publications have demonstrated that DL could reduce metal artifacts significantly through projection completion,[8–10] image post-processing,[16–21] and dual-domain processing.[22–28] Despite decades of research in MAR, a universal benchmark for objectively comparing MAR techniques remains absent. CT MAR developments often rely on their own test cases and evaluation metrics, without comprehensive performance comparisons with other MAR methods. Some past MAR studies used specific clinical cases or customized phantoms containing target metal implants for evaluation, as they are designed with a particular application in mind. These phantoms and clinical cases include pelvis with hip prostheses, head with dental fillings, and thorax with spinal stabilization rods. Common evaluation metrics include CT number accuracy, metal diameter accuracy, and the severity of streak artifacts by thresholding.[7,31,34,35] Many deep learning-based MAR studies rely on numerical simulations, where metal artifacts are synthetically introduced into randomly selected clinical CT images. The simulation settings—such as geometry and metal properties—vary across studies. Popular quantitative evaluation metrics are RMSE, PSNR, and SSIM.[21,24,36] There is a significant need for a universal CT MAR image quality benchmark to assess the clinical impact of new MAR methods and compare them to state-ofthe-art techniques. 

Med Phys. Author manuscript; available in PMC 2026 January 02. 

Haneda et al. 

Page 4 

The goal of the AAPM CT metal artifact reduction (CT-MAR) challenge was to create and distribute a clinically representative 2D MAR performance benchmark under transparent and unbiased conditions, and to invite participants to objectively compare the performance of their MAR methods based on this benchmark. We provided realistic large-scale training datasets including a wide variety of anatomies and metals in both image- and projectiondomains, along with reconstruction and reprojection tools, allowing participants to work in either or both domains. For evaluation, we designed scoring datasets based on different metal implant scenarios using head, abdomen, and pelvis regions. The metals were carefully designed and manually inserted to meaningful locations. A scoring tool was developed based on eight image quality scoring metrics, and region-of-interest (ROI) was carefully selected for each case to quantify metal artifacts. The benchmark includes a number of intermediate scores for various image quality metrics and also a single average score summarizing the overall performance of a specific MAR approach. 

While MAR solutions could generally involve dual-energy CT or photon-counting CT, our scope was limited to single-energy CT to ensure broadest applicability (single-energy MAR can also be adopted for dual-energy MAR). We made the conscious decision to focus on 2D MAR approaches only, since the vast majority of MAR research is performed in 2D and in some cases later extended to 3D. Nevertheless, the physics modeling was realistic, extracting only the center row of a 3D CT scanner, but still including scattered radiation of a wide-cone 3D CT geometry. 

The AAPM CT MAR challenge[37,38] was organized by GE HealthCare Technology and Innovation Center, Massachusetts General Hospital, and Rensselaer Polytechnic Institute from October 2023 to July 2024. The top three teams shared in an award pool of $4000: $2000 for the first place, $1500 for the second place, and $500 for the third place (sponsored by GE HealthCare and First-imaging Medical Equipment). Additionally, the two top-performing teams and the organizer were invited to the 2024 AAPM Annual Meeting & Exhibition in Los Angeles to present their methods and a summary of their work.[39] This Grand Challenge was endorsed by the Medical Image Computing and Computer Assisted Intervention Society (MICCAI). 

This paper is organized as follows: Section 2 describes the methods including grand challenge structure (2.1), training datasets (2.2), scoring datasets and evaluation metrics (2.3), and additional test datasets and metrics (2.4). Sections 2.5 to 2.7 briefly introduce the MAR algorithms of the top three teams. Section 3 presents the algorithm survey results (3.1), score distributions (3.2), and sample qualitative image results (3.3). Finally, the paper concludes with a discussion (Section 4) and conclusion (Section 5)). 

## **2 | METHODS** 

## **2.1 | Grand challenge structure** 

The AAPM CT MAR grand challenge invited participants to develop a 2D metal artifact reduction algorithm. The challenge consisted of three phases:Phase 1: Training & Development phase (Oct 30,2023–Feb 18,2024), Phase 2: Feedback & Refinement phase (Feb 19, 2024–May 5, 2024), Phase 3: Final Scoring phase (May 6, 2024–May 20, 2024). 

Med Phys. Author manuscript; available in PMC 2026 January 02. 

Page 5 

Haneda et al. 

Figure 1 illustrates an overview of the grand challenge structure. In Phase 1, training datasets were provided to the participants for their MAR algorithm development. In Phase 2, five clinical datasets were provided to the participants for the preliminary scoring. In Phase 3,29 clinical datasets were provided, and the final scores and ranking were computed using the scoring metrics. 

During Phase 1 (training and development phase), 14,000 CT training datasets were provided. They were generated using the CatSim CT simulator in the open-access toolkit XCIST,[40,41] employing a hybrid data simulation framework that combines publicly available clinical images[42,43] and virtual metal objects. The training dataset includes five types of data: CT sinograms (with and without metals), CT reconstructed images (with and without metal artifacts), and binary metal masks as shown in Figure 2(1)–(5). Participants could choose to develop image-domain, sinogram-domain, or dual-domain methods. For example, an image-domain method would generate (1) directly from (2). A sinogram-domain method would estimate (3) from (4) and then use the reconstruction routine to estimate (1). A dualdomain method would start from (4), perform sinogram-domain operations, reconstruction, and image-domain operations to achieve (1), possibly within an iterative framework. A standard 2D filtered back projection (FBP) reconstruction routine and reprojection routine were provided[38] in Python so that participants did not need to develop their own forwardand back- projection methods. 

During Phase 2 (Feedback & Refinement phase), a total of five clinical uncorrected datasets were provided in sinogram domain and in image domain for preliminary scoring (Figure 2 (2) and (4)). No ground truth images were provided. Participants could process the data using their MAR algorithms and submit their image results through the challenge website to see their preliminary scores and rankings on the leaderboard. The preliminary score was computed using eight image quality metrics (Section 2.3). The metal implant scenarios, mathematical definitions of metrics, and thresholds for metal integrity were provided to the participants.[44] The ROIs were not provided. Participants were allowed to submit their results up to three times during Phase 2. 

During Phase 3 (Final Scoring), a total of 29 clinical uncorrected datasets were provided in both sinogram and reconstructed image formats for the final scoring. No ground truth sinograms or images were provided. Participants were given 2 weeks to submit the final results. Datasets covered the most relevant clinical scenarios, including surgical clips, dental fillings, and hip prostheses. The metal implants were inserted into clinically realistic locations. As in Phase 2, eight image quality metrics were computed for each case. The final scores and rankings were calculated based on the average across all cases and metrics. Participants were asked to submit an image without any metal artifacts but still containing metal object itself, as shown in Figure 2 (6). Since the final image should still contain the metal object, participants needed to estimate the contour of the metal object and reintroduce it in case the metal was removed during the processing. 

## **2.2 | Training datasets** 

A previous publication[2] provides a detailed description of the training data and evaluation metrics, so we provide only a brief overview here. In deep learning, the performance heavily 

Med Phys. Author manuscript; available in PMC 2026 January 02. 

Page 6 

Haneda et al. 

depends on the quality of the training datasets. To generate realistic CT data with metal artifacts, we used a hybrid data simulation framework based on the CatSim CT simulator in the open-access toolkit XCIST.[40,41] This hybrid framework combines clinical patient images and user-defined virtual metal objects as input to the simulation tool. The clinical patient images were selected from publicly available CT database. 12,227 “body” images with different anatomies (lung, abdomen, liver, pelvis, and head images) were collected from about 1,230 patients in the NIH DeepLesion dataset.[42] Additionally, 1,B773 “head” images were collected from 10 patients in the UCLH Stroke EIT Dataset.[43] Patients scans with very small metal objects (≤30 pixels) were included in the training data, while those with more extensive metal objects were excluded, to avoid pre-existing streaks in the ground-truth images prior to introducing virtual metal objects. We provided a list of cases with small metal objects so that the affected slices could be excluded by participants if desired. Moderate spatial frequency boosting[45] was applied to the patient images prior to CT simulation to compensate for the spatial blur associated with simulation and reconstruction. As a result, simulated and reconstructed images without virtually inserted metal objects were visually indistinguishable from the original clinical images, which is an important property of the hybrid data simulation framework. 

Virtual metal objects were defined as random shapes synthesized by generating vertices of a random fractal shape. The metal objects were inserted in random soft tissue or bone locations. The metal materials included amalgam, stainless steel, copper, cobalt, and titanium. Up to five metal objects of the same material with different shapes were inserted per image. Datasets containing no metal were simulated as “label”datasets (i.e., ground truth). To facilitate localization of the metal objects, we generated a metal mask image with the same dimensions as the reconstructed images. More details on insertion of virtual metal objects can be found in our previous publication.[2] 

CatSim models the X-ray beam, its interaction with matter, and the X-ray detection process based on X-ray physics. Because simulation of X-ray attenuation depends on the materials, a database of material attenuation properties is defined including tissue/water and different metal types. The X-ray source spectrum is defined as a sum over a discrete set of energy levels (i.e., 12 energies). Signal attenuation along each path from source to detector is calculated at each energy level, and then summed over energy levels to yield a single measurement at the detector. 

We used a nominal vendor-neutral CT geometry (not exactly matching any commercial CT scanner) for the simulations using the following parameters: source-to-iso-distance 550 mm, source-to-detector distance 950 mm, 1.0 mm × 1.0 mm focal spot size using a VCT scanner profile, 1.0 mm × 1.0 mm detector cells with fill factor 90% × 90%, 1.25 detector quarter offset, 120 kVp tube voltage, 500 mA tube current, one detector row, 900 detector columns, 1000 views, a large bowtie filter, and using realistic quantum noise and electronic noise. To realistically reflect finite focal spot size, detector cell size, and rotational blur, the X-ray beam was simulated as the sum of multiple projection lines. More specifically, the focal spot width, focal spot length, and gantry rotation were each sampled with two sample points, resulting in eight sub-samples total. A distance-driven projector was used to model the finite detector cell size. 

Med Phys. Author manuscript; available in PMC 2026 January 02. 

Haneda et al. 

Page 7 

The forward scattering process, taking into account scatter rejection by an anti-scatter grid, is modelled as a modified convolution-based approach,[46] where a primary signal is convolved with Monte-Carlo derived kernels. We assumed an anti-scatter grid with an aspect ratio of 9.8, resulting in a scatter kernel width of 49 detector columns. This was repeated for all energies to consider the energy-dependence of the scatter kernel, and a weighted sum of scatter contributions was used. To minimize computation time, scatter was computed for a single detector row and scaled to mimic scatter for 64 detector rows, representing the physics aspects of a realistic 3D CT scenario. The images were reconstructed by Feldkamptype filtered backprojection using a standard reconstruction kernel. The reconstruction fieldof-view (FOV) for body data was set to 400 mm, and the one for head was 220.16 mm over 512 × 512 pixels. Water beam hardening correction and simple kernel-based scatter correction were applied to all datasets. 

We previously evaluated the modeling accuracy of CatSim in terms of spectrum shape,[47 ] spatial resolution,[48] quantum noise, electronic noise, and scatter. With the calibrated parameters, we observed good agreement in metal artifact appearance between simulation and measurements. The details of our CT physics model and the validation of metal artifact simulation can be found in our previous publication.[2] 

Figure 3 shows examples of training datasets generated with our hybrid data simulations from real patient images and virtual metal objects. The top row is a body (abdominal) data with three stainless steel objects with 9.3, 5.2, and 2.8 mm in diameter. The bottom row is a head example with three amalgam objects with 3.1, 2.1, and 1.5 mm in diameter. For each row, five types of data corresponding to Figure 2(1)–(5) are shown: two sinograms (without and with metal objects), two reconstructed images (without and with metal objects), and a metal mask. Although ground truth data are part of the training datasets, it was up to the participants to decide whether to use a supervised or an unsupervised approach. Similarly, the participants could use DL or non-DL approaches. 

## **2.3 | Scoring datasets and metrics** 

The scoring datasets were based on clinical CT scans acquired at the Massachusetts General Hospital. Unlike the training datasets, where metal objects were inserted at random locations, here metal objects were manually designed and positioned to represent realistic anatomy and metal combinations (e.g., gold markers for radiotherapy positioning in the prostate). The final scoring dataset contains a total of 29 clinical cases with the clinically most relevant metal scenarios,[2] covering all categories of artifacts from minor shading to substantial degradation, as defined by Gjesteby et al.[1] The scenarios include small sized metal objects (surgical clips, fiducial marker seeds, and dental fillings), medium sized objects (pacemakers, larger dental work, and spinal reconstruction) up to large, full joint replacements (shoulder and hip). The patient datasets covered both male and female patients from different age groups. The final scoring datasets were generated by the CatSim simulator using the same CT scanner geometry as the training datasets. The preliminary scoring dataset for Phase 2 was generated in the same fashion but there was no overlap between the Phase 2 datasets and the final scoring datasets. 

Med Phys. Author manuscript; available in PMC 2026 January 02. 

Haneda et al. 

Page 8 

A total of eight scoring metrics were computed to evaluate MAR quality on the scoring datasets. The metrics included CT number (CTN) accuracy, noise, image sharpness, streak amplitude, structural integrity (SSIM), metal integrity, bone integrity, and the influence on proton beam range for radiotherapy (PBR). Each scoring metric was normalized to a range from 0 (no relevant differences relative to ground truth) to 4.0 (no improvement over uncorrected image or our defined worst case). The score was calibrated by assigning a score of 2 to the popular NMAR[7] algorithm, where applicable. The total score was computed as the average of the eight scoring metrics. A short summary of each metric is provided here, and a full description is given in the previous publication.[2] 

CT number accuracy (CTN) was defined as the root-mean-square error (RMSE) between the ground truth and the MAR image within the patient, excluding the metal ground truth geometry. Noise was defined as the standard deviation within a circular ROI in homogenous soft tissue that is least affected by the metal artifacts. A linear increase from the ground truth noise (score 0) to roughly two times the ground truth noise (score 4) was used for normalization. Image sharpness was quantified as the preservation of gradients within specified ROIs containing an edge between soft tissue and bone as well as between soft and adipose tissue. A Sobel filter was applied to determine the absolute gradient magnitude within the respective ROI. Sharpness was then quantified as the 90th percentile gradient magnitude value to minimize the influence of outliers, and then averaged over the two regions. Then, it was normalized so that a score of 0 corresponds to an unaltered ground truth, whereas for a score from 1 to 4, a Gaussian blur with a sigma of 0.4, 0.5, 0.75 and 1 is introduced, respectively. Streak amplitude was assessed within ROIs perpendicular to strong streaks in the uncorrected images. To minimize the effect of outliers, the average over the highest and lowest 5% CT number deviation to ground truth in each ROI was calculated, respectively. The streak amplitude was then defined as the difference between the two. Structural integrity is quantified using the structural similarity index (SSIM) within the patient geometry excluding the metal ground truth. For this, the implementation by Wang et al.[49] was followed. In Bone integrity, all voxels with a CT number above 150 HU (excluding the metals) were considered bone. The bone integrity was determined by the average of two values: volume change and Sørensen–Dice coefficient between the ground truth and estimated bone in the corrected images to assess the similarity. Metal integrity was evaluated analogous to the bone integrity. To quantify the metal integrity, all voxels above a certain threshold, defined as the highest CT number in the ground truth without metals plus a margin of 250 HU, were considered metal. The analysis was limited to a ROI covering the metal ground truth and the surrounding tissue to exclude noise. The metal threshold for each test case was provided to the participants. We recommended the participants to fill in their estimated metal region with the value at or slightly above the threshold. Proton beam range (PBR) is a metric used in radiotherapy. There, CT numbers are translated into the tissues’ different stopping power relative to water (SPR). The SPR multiplied with the pixel size corresponds to the water-equivalent thickness (WET) of the tissue. Thus, by being integrated along a beam path, the PBR is calculated. To assess the effect on clinical treatment accuracy, the clinically validated CTN-to-SPR translation curve from Massachusetts General Hospital was applied to the CT images. Accuracy was then quantified for simplified in-plane beam paths in comparison to the artifact-free ground truth image. A score of zero corresponds to 

Med Phys. Author manuscript; available in PMC 2026 January 02. 

Haneda et al. 

Page 9 

no WET shift, linearly increasing to a score of 4 for a range shift of 2% relative to the largest beam range of the treatment field. The description of all 8 scoring metrics was published for the participants before Phase 3. The respective ROIs for analysis were illustrated in the supplemental document of our previous publication.[2] A python version of the benchmark tool is made available online via GitHub.[38] 

## **2.4 | RMSE/PSNR/SSIM datasets** 

In addition to the scoring datasets, participants were provided 1,000 new test cases to analyze the performance of various algorithms in a larger number of cases (850 from Deep Lesion datasets and 150 from UCLH Stroke EIT dataset) but based on simpler metrics. We asked the participants to process the cases with their MAR and compute RMSE/SSIM/PSNR values using our script. These datasets were independent from the scoring datasets. The results were not included in the final score but are illustrative for the algorithms’ behavior in terms of these standard metrics. 

## **2.5 | MAR approach #1 (first place)** 

The MAR algorithm proposed by the first-place team (including authors Yi Guo and Jianhua Ma from Southern Medical University and Xi’an Jiaotong University) is based on a dual-learning framework. Traditional deep learning architectures predominantly focus on unidirectional mapping from degraded sinograms to reconstructed images,[50,51] which frequently leads to compromised reconstruction accuracy and stability. To overcome this inherent limitation, a novel dual-learning-based closed-loop framework has been proposed, establishing dual cycle consistency constraints between reconstructed images and original sinograms. The proposed framework simultaneously optimizes two mappings: a primal mapping that transforms metal-contaminated sinograms to artifact-free images, and a dual mapping that transforms the images back to the original sinograms. Additionally, the physical mechanisms that produce metal artifacts are explicitly integrated into both mappings. The implementation of the dual cycle and physical mechanism significantly enhances the robustness and precision of the reconstruction outcomes. While the training phase necessitates the joint optimization of both primal and dual mappings, the inference stage only requires the application of the primal mapping for image reconstruction. The application of dual learning in CT reconstruction can be referenced to our their studies.[52] 

The overview of the proposed algorithm is illustrated in Figure 4. The primal mapping integrates four specialized correction modules for photon starvation, beam hardening, noise, and scatter to generate artifact-free reconstructions. Conversely, the dual mapping models these physical mechanisms to estimate the corresponding sinogram from the reconstructed image and calculate the consistency loss with the original sinogram. In the primal mapping, the photon starvation effect is treated as an attenuation truncation problem, correcting it by substituting truncated data with a projection of an estimated prior image, that is obtained by restoring the uncorrected FBP image using a convolutional neural network. Beam hardening is corrected using a multi-layer perceptron network in the primal mapping. In the dual mapping, it is modeled through multi-energy forward projection. More specifically, a convolutional neural network is utilized to estimate the density map of the reconstructed images. Then, reprojection operation is subsequently performed on the estimated density 

Med Phys. Author manuscript; available in PMC 2026 January 02. 

Haneda et al. 

Page 10 

map, and multi-energy projection data corresponding to the reconstructed image are calculated by incorporating X-ray spectrum and material-specific attenuation profiles. To remove noise and scatter signals, an image domain network is utilized to further suppress residual artifacts in the primal mapping. In the dual mapping, poly-energetic projection is used with noise and scatter models learned by a projection domain network. 

The U-Net architecture is utilized in the experiment. To prevent gradient vanishing during the joint training of the entire model, all modules employ self-built differentiable modules. By integrating deep learning based physical models, this model outperforms other approaches that directly forward-project the reconstructed image and calculate the consistency loss with the original sinograms because original sinograms are degraded by factors such as beam hardening and photon starvation. Even when reconstructed images are accurate, the projection data estimated through direct forward projection remains inconsistent with the original sinograms, failing to effectively constrain network training. The proposed algorithm models the physical mechanism of metal artifacts during the forward projection, thereby addressing this issue. 

To ensure effective model training, an additional 15,000 head images collected from local hospitals and simulated with five types of material. The Mean Squared Error (MSE) loss function and Adam optimizer were employed to optimize the framework with the parameters ( β 1, β 2) = (0.9, 0.999). 

## **2.6 | MAR approach #2 (second place)** 

The MAR algorithm proposed by the second-place team (including authors Hyoung Suk Park and Kiwan Jeon from National Institute for Mathematical Sciences in Daejeon) is based on the same principle as the Normalized Metal Artifact Reduction (NMAR) approach[7 ] and combines it with advanced DL steps. In NMAR, the key component is a prior image used for inpainting. By leveraging the advanced representational capabilities of implicit neural representations (INRs),[53–55] particularly through the use of a multi-layer perceptron (MLP) network, the proposed method generates a prior image using metal-unaffected projection data. In the NMAR procedure, the projection data are normalized by the forward projection of the prior image, and linear interpolation is employed to remove the metal-affected projection data along the metal trace. The inpainted projection data are subsequently reconstructed using the FBP method. The resultant NMAR image contains secondary artifacts caused by inaccuracies in interpolation and prior image generation. To further suppress these secondary artifacts in the NMAR image, residual learning is applied by incorporating metal features into the residual network. The schematic diagram of the proposed method is illustrated in Figure 5a. 

## **2.6.1 | INR-based prior image generation:** The prior image is implicitly generated 

by a trainable MLP network, where the input is 2D positions within the image domain, and the output corresponds to the attenuation coefficients. In accordance with the Lambert-Beer law, the MLP network is trained to ensure that the line integral of the output values along the X-ray lines matches the projection data. The L1 loss between the line integral and the projection data is evaluated in regions outside the metal traces, thereby enabling the network 

Med Phys. Author manuscript; available in PMC 2026 January 02. 

Haneda et al. 

Page 11 

to generate a CT image prior that excludes metal artifacts. The proposed network consists of five fully connected layers, each comprising 64 nodes. The sinusoidal activation function[56 ] is employed for all layers, while the output layer uses a linear activation function. The initial weights for the network are established using a meta-learning framework.[57] More specifically, the network is pre-trained on 5,000 ground-truth projection datasets randomly sampled from a total of 14,000 projection datasets provided in Phase 1. The weights obtained from this pre-training process serve as initial parameters to facilitate prior image generation during the test phase. The network is trained using the Adam optimizer with a learning rate of 5 × 10[−4] . After each update step of the network, negative outputs of the network are clamped to zero to enforce non-negativity, improving the network’s stability.[55] 

## **2.6.2 | Residual learning with metal feature embedding:** The residual network to 

reduce secondary artifacts in NMAR images takes the NMAR image and the segmented metal mask as inputs. The output represents the residual between the ground truth and the NMAR image. The network design is motivated by the observation that secondary artifacts in the NMAR image are closely linked to the metal structures, making it essential to incorporate metal-related information into the network. The network architecture is illustrated in Figure 5b. The network is constructed with an encoder-decoder structure and skip connections, where the metal mask is fed into the network through multiple convolutions and fully connected layers, followed by an adaptive instance-normalization (AdaIN) layer. A detailed explanation of the AdaIN operation is provided in our previous study.[58] To address the domain discrepancy between training and test domains, the residual learning process is performed in a patch-by-patch manner. 

In summary, the proposed method consists of the following steps: [Step 1] Segment the metal mask from the uncorrected CT image using a simple thresholding method.[Step 2] Perform a dilation operation on the segmented metal mask, followed by forward projection to obtain the metal trace in the projection domain. [Step 3] Generate the CT image prior using the MLP network trained on the projection data from regions excluding the metal traces. [Step 4] Apply the NMAR process using the generated CT image prior. [Step 5] Perform secondary artifact correction on the NMAR image using the residual correction network that takes the NMAR image and segmented metal mask as inputs. [Step 6] To correct only the metal-affected projections, reapply the NMAR process by using the residual-corrected CT image obtained from Step 5 as a new CT image prior. [Step 7]. Combine the metal mask obtained from Step 1 with the corrected CT image obtained from Step 6, producing the final artifact-corrected CT image. 

## **2.7 | MAR approach #3 (third place)** 

The proposed method by the third-place team (including authors Fuxin Fan and Mareike Thies from Friedrich-Alexander-Universität Erlangen-Nürnberg) uses a cross-domain framework. The entire pipeline of the proposed framework is illustrated in Figure 6. It consists of six steps and operates on both the sinogram and image domains. The first step, metal segmentation, is critical as it lays the foundation for the subsequent processes. This is accomplished using a dual-domain segmentation network, which integrates two interconnected Swin UNETR networks.[59] These networks are linked by a differentiable 

Med Phys. Author manuscript; available in PMC 2026 January 02. 

Haneda et al. 

Page 12 

forward projection operator, implemented using the Operator Discretization Library[60 ] (ODL). The process begins with the input of the sinogram containing metal traces into the first network, which outputs a metal mask in the sinogram domain. This mask is then reprojected onto a flat detector configuration by rebinning from the original circular detector configuration and backward projected into the image domain using ODL. With the results from the previous network, the second network processes the metal-affected image to produce a corresponding metal mask in the image domain. Two Dice loss functions are applied to optimize segmentation in both domains. Following segmentation, the consistency check step ensures the accuracy of the segmented sinogram by minimizing false positives and false negatives.[61] 

In the second step, the linear interpolation method is applied to the sinogram to produce an inpainted version, from which the corresponding reconstructed image is generated. In the third step, an image-based DICDNet[16] is used for an initial round of metal artifact reduction, producing an artifact-reduced image. The DICDNet used in the framework consists of 10 stages of X-Net and M-Net, with losses calculated between the outputs of X-Net and M-Net from all stages. In the fourth step, starting with the artifact-reduced image, a forward projection of this image is used to guide the network-based projection inpainting process. An inpainting network, built on the Swin UNETR architecture, generates a more consistent sinogram compared to the previous linearly interpolated sinogram. Leveraging the improved reconstruction from the updated sinogram, the same DICDNet is applied once again to produce an enhanced image. The final output image is obtained by overlaying the metal mask onto that enhanced image, effectively restoring the image with reduced artifacts. 

## **3 | RESULTS** 

## **3.1 | Algorithm survey** 

A total of 106 teams (participants) registered for the CT-MAR Challenge, with 26 teams completing all phases and submitting final results. Of these, 34% were from institutes in the United States, 23% from South Korea, 23% from China, and 8% from Germany. The remaining teams were from Finland, the UAE, and Japan. 77% of the teams were from academic institutes and the remaining teams were from medical institutes or industry. Table 1 summarizes the final score of participants and the algorithm survey we conducted. The algorithm survey included the domain of the MAR algorithm (image or sinogram), the use of training data, the metric(s) to optimize the MAR approach, the choice of DL architectures, and a list of related publications. The table is ordered by rank based on the final average score, an average across the 29 cases and 8 metrics, which were normalized to a scale of 0.0 to 4.0 (0: good, 4:bad). For comparison, the traditional NMAR[7] approach is also listed at the bottom. NMAR is a non-deep learning (non-DL) inpainting method, where the metal trace region is filled by linear interpolation in a normalized sinogram domain.[7] 

In total, 27% of the teams used sinogram domain approaches, 31% used image domain approaches, and 42% worked in both domains. Most teams used the provided training datasets, except for those who used a non-DL approach. Two teams added more head datasets to compensate for the imbalance in the training datasets. Almost all teams used CT 

Med Phys. Author manuscript; available in PMC 2026 January 02. 

Haneda et al. 

Page 13 

number accuracy or equivalent metrics in the loss function to optimize their algorithms, with many also including other defined scoring metrics. 

According to the survey, 92% of the teams used Deep Learning (DL) approaches and 8% used analytical (non-DL) approaches. Figure 7 shows the statistics of the DL types selected by the participants. The participants chose a variety of network architectures, which we categorized into six families: UNet,[62] ResNet,[63] GAN,[64] Diffusion model,[65,66 ] Transfomers,[67] and others. Note that most of the teams used more than one network architecture to develop their MAR. All network architectures declared by the participants in the survey are included in the Figure 7 and listed in Table 1. The highest-ranking non-DL approach placed 15[th] so all top 14 teams used DL approaches. The diffusion model, a relatively new generative network, was one of the popular choices in this challenge. 31% of the teams used a diffusion model, with the highest-ranking team in 5[th] place and 4 teams in the top 10. Transformers were another popular network architecture, used by 19% of the participating teams, with the highest ranking team in 4[th] place and 4 teams ranked in the top 10. The highest-ranked image-domain-only approach was in 4[th] place, while the highest-ranked sinogram-domain-only approach was in 14[th] place. The top 3 teams used dual domain approaches, combining the strengths of both domains. 

## **3.2 | Scores** 

While Table 1 shows the final score of each participant, Figure 8 illustrates the distribution of each scoring metric. Most metrics showed a broad distribution, with the exception of noise and sharpness, which were biased toward 0 (‘good’) because we adjusted the score normalization to ensure that excessive denoising or sharpening wouldn’t provide an advantage. Since denoising and sharpening were not the focus of this challenge, good scores were assigned in most cases unless extremely large noise was observed. Table 2 shows a more detailed breakdown of scores from selected participants (in 1[st] , 2[nd] , 3[rd] , 10[th] , and 21[st ] place) as well as the NMAR algorithm. 

Figure 9 shows the additional results of RMSE, SSIM, and PSNR on the 1,000 MAR processed images by participants. The MAR processed images were clipped to the range of [−2000 6000] HU and normalized to the [0 1] range before metric calculation. The area outside of the scan field-of-view (FOV) was excluded from the calculation. Three participants (#8, #22, and #23) were excluded from the plot due to the large bias and variance for displaying purpose. Overall, RMSE increases for lower rankings, while SSIM and PSNR decreases for lower rankings, as expected. 

## **3.3 | Sample image results** 

Figure 10 shows the MAR processed images by the top 3 participants, as well as images from some representative participants in the 10[th] and 21[st] place, alongside NMAR, uncorrected, and ground truth images for head with data fillings, thorax with spinal screws, and pelvis with a prosthesis. Figure 11 shows the ROI locations used for the evaluation. Note that the labeled rankings are based on the final score averaged across all test cases, not based on the individual test cases shown. The images from the top ranked teams showed outstanding metal artifact reduction, including very good streak suppression and tissue 

Med Phys. Author manuscript; available in PMC 2026 January 02. 

Haneda et al. 

Page 14 

restoration. As the ranking decreases, metal artifacts become more apparent, as reflected by the scoring. Some of the very low-ranked images showed no improvement, CT number range mismatch, or even patient size mismatch. 

We observed that each MAR algorithm has different strengths for artifact suppression, as reflected in Table 2, where the performance of each team varied between the different test cases. The MAR approach #1 (first place) showed the best CT number (0.74), low noise (0.01), streak suppression (1.29), accurate SSIM (0.75), and bone integrity (0.89) as shown in Table 2. This can be verified in Figure 10. For example, in the pelvis case, the streaks from the hip prosthesis were successfully eliminated, while the CT number accuracy in the surrounding tissues, including the urinary bladder region, was well preserved compared to other algorithms (red arrow). The algorithm did not suffer from CT number bias or distortion in regions affected by metal artifacts. The contours of the hip prosthesis were well restored and showed positive agreement with the ground truth. The MAR approach #2 (second place), on the other hand, showed improved detail preservation with sharpness (0.52) in Table 2. Especially in uniform tissue regions, the texture was more similar to the ground truth than other algorithms, as indicated by the yellow arrows in Figure 10. However, this came at the expense of increased streaks (1.65) and noise (0.19), as shown by the blue arrows. The MAR approach #3 (third place) had moderate artifact suppression (1.48), low noise (0.01), and good sharpness (0.54). In some cases, higher CT numbers than the ground truth were observed in streak-affected regions, as shown in the pelvis case in Figure 10. The MAR approach #10 (tenth place) showed the most noise (0.44) and streaks (2.15). The MAR approach #21 (twenty-first place) had the largest overall CT number error (1.72) among the five teams. Yet, all of these approaches either outperformed or were comparable to NMAR in terms of overall score. 

## **4 | DISCUSSION** 

106 teams from around the world registered for the AAPM CT-MAR grand challenge and 26 teams submitted their final results in phase 3. This is a high level of participation considering that MAR research requires in-depth knowledge of CT acquisition and reconstruction in addition to the MAR algorithms themselves. The submissions were of high quality, with more than 70% of the teams achieving a better total score than the basic version of the popular NMAR method. The top teams demonstrated outstanding metal artifact reduction, particularly through effective streak suppression, and minimized the distortion in affected regions. The performance of the top 5 algorithms in the final ranking was stable and robust, with their individual cases ranking within the top 10 in 90% of cases. A side effect of deep learning methods is hallucinations. In regions contaminated by metal artifacts, deep learning methods sometimes restores tissues differently from the ground truth with their own estimations. For example, heart region and urinary bladder region are restored with different shading patterns by some algorithms in Figure 10. Guiding deep learning toward accurate restoration or developing methods to rate the uncertainty could be a focus for future MAR research. 

Note that methods trained with simulated metal artifacts generally can be applied directly to data with real metal objects. This is particularly true for projection-domain methods 

Med Phys. Author manuscript; available in PMC 2026 January 02. 

Page 15 

Haneda et al. 

that consider metal-corrupted data as missing, and for image-domain methods if the simulated artifacts are shown to be very realistic. However, methods based on physicsbased corrections may be more sensitive to small deviations of the physics models in the simulations from the true CT scanner physics. We expect retraining and tuning is required for dealing with the domain shift, particularly when applying to different CT scanner models or scan protocols. 

Through this challenge, we established a framework to benchmark MAR algorithms using training data and a scoring tool. We acknowledge several limitations associated with the study. First, our training data contained a small percentage of head data due to the limited availability of public head data. In particular, there were relatively few images containing dental implants and fillings. This could result in a bias in MAR performance such as poor MAR in dental regions and excellent MAR in body regions. Second, our challenge was limited to 2D MAR. Although we had good reasons to limit this challenge to 2D, this approach could be extended to 3D multi-slice CT for more rigorous evaluation of MAR algorithms that are closer to clinical deployment. Third, although our scoring metrics cover general image quality metrics, more refinements could be made from a diagnostic point of view. For example, while some metrics are tailored to specific applications (e.g., PBR), the benchmark could include other application-specific metrics. Also, task-based metrics such as lesion detectability are still lacking and would be an important additional way to evaluate the effective clinical performance of various MAR approaches. However, note that when changing the benchmark, scores can no longer be compared to the scores generated with the current benchmark. 

Another limitation is that the original ground truth originates from public datasets, which already contain some noise and possibly artifacts. We carefully avoided including any metal artifacts in the ground truth images. Although the noise and streak levels in these images were substantially lower than those in the simulated metal artifact images, the ground truth is not entirely free of artifacts. Additionally, we fixed the reconstruction voxel size to simplify the development and comparative evaluation. This may not reflect clinical practice, where different voxel sizes may be used depending on the anatomical site. Our intuition is that only image-domain methods are sensitive to voxel size, and that voxel size itself is not a fundamental factor in the success of a given MAR approach. Yet, image-domain MAR methods may need to be retrained and adapted to generalize across varying voxel sizes. Finally, it is worth noting that the metal trace in the training sinogram generated by the XCIST simulator may look distorted for large metal objects with diameters larger than 3.0 cm. This is due to the imperfection of the kernel-based scatter correction, which was typically tuned for tissue/water and is less effective for large metals. Out of the 14,000 datasets, 274 were affected. These datasets can be excluded from the training process using the metal mask diameter information provided in the training datasets. 

We want to highlight that all training datasets, scoring datasets, scoring tools, and development tools have been made available via GitHub[38] after completion of the grand challenge. MAR researchers can download our 14,000 2D MAR training datasets for training their own new MAR methods. Furthermore, we have created and distributed a MAR benchmark enabling MAR researchers to objectively and comparatively evaluate their new 

Med Phys. Author manuscript; available in PMC 2026 January 02. 

Page 16 

Haneda et al. 

approaches using the same metrics used in the CT MAR grand challenge. The best overall score achieved so far was 0.96, but we hope that future MAR research will lead to even more powerful approaches, resulting in even better overall scores using this same CT MAR benchmark. By providing publicly accessible tools and datasets, the challenge has facilitated ongoing research in the field and offers a valuable resource for the further development of MAR solutions. 

## **5 | CONCLUSION** 

In this paper, we summarized the AAPM CT Metal Artifact Reduction (CT-MAR) grand challenge. The goal of the CT-MAR challenge was to provide a clinically representative benchmark for evaluation of CT MAR algorithms. Through this challenge, we developed a hybrid training data generation method by simulating a wide variety of metals and resulting artifacts using publicly available clinical images. 106 teams from around the world registered for the AAPM CT-MAR grand challenge. Of these, 26 teams submitted results in the final phase of the grand challenge. Their submitted results were evaluated based on our clinically relevant metrics. The grand challenge was highly competitive, featuring diverse methods and a wide variety of results, including some truly exceptional results. After the challenge, the training datasets and the MAR benchmark tool were publicly released[37] to support future MAR development and comparative evaluation. 

## **ACKNOWLEDGMENTS** 

We would like to thank Karen Drukker, Emily Townley, Emil Sidky, and the AAPM Working Group on Grand Challenges (WGGC) for all support. We also would like thank Benjamin Bearce and Upasana Thakuria for the support on the challenge website platform. We are also grateful to the participants for their efforts. We would like to thank Yongbo Wang from Xi’an Jiaotong University, Zhaoying Bian and Dong Zeng from Southern Medical University, and Andreas Maier from Pattern Recognition Lab, Friedrich-Alexander-Universität Erlangen-Nürnberg to introduce their algorithms as the top 3 winning teams. Lastly, we would like to thank GE HealthCare and Firstimaging Medical Equipment for sponsoring this challenge. Research reported in this publication was supported by the NIH/NIBIB grant R01EB031102. The content is solely the responsibility of the authors and does not necessarily represent the official views of the NIH. 

## **DATA AVAILABILITY STATEMENT** 

All training datasets, scoring datasets, scoring tools, and development tools have been made available via GitHub.[38] 

## **REFERENCES** 

1. Gjesteby L, De Man B, Jin Y, et al. Metal artifact reduction in CT: where are we after four decades?. IEEE Access. 2016;4:5826–5849. doi:10.1109/ACCESS.2016.2608621 

2. Nils P, Haneda E, Zhang J, et al. A hybrid training database and evaluation benchmark for assessing metal artifact reduction methods for imaging and therapy. Med Phys. 2025. doi:10.1002/mp.70020 

3. Kohyama S, Yoshii Y, Okamoto Y, Nakajima T. Advances in bone joint imaging-metal artifact reduction. Diagnostics. 2022;12(12):3079. doi:10.3390/diagnostics12123079 [PubMed: 36553086] 

4. De Man B, Nuyts J, Dupont P, Marchal G, Suetens P. Metal streak artifacts in X-ray computed tomography: a simulation study. IEEE Trans Nucl Sci. 1999;46(3):691–696. doi:10.1109/23.775600 

5. Park HS, Hwang D, Seo JK. Metal artifact reduction for polychromatic X-ray CT based on a beam-hardening corrector. IEEE Trans Med Imaging. 2016;35(2):480–487. doi:10.1109/ TMI.2015.2478905 [PubMed: 26390451] 

Med Phys. Author manuscript; available in PMC 2026 January 02. 

Haneda et al. 

Page 17 

6. Mehranian A, Ay MR, Rahmim A, Zaidi H. X-ray CT metal artifact reduction using wavelet domain L0 sparse regularization. IEEE Trans Med Imaging. 2013;32(9):1707–1722. doi:10.1109/ TMI.2013.2265136 [PubMed: 23744669] 

7. Meyer E, Raupach R, Lell M, Schmidt B, Kachelrieß M. Normalized metal artifact reduction (NMAR) in computed tomography. Med Phys. 2010;37(10):5482–5493. doi:10.1118/1.3484090 [PubMed: 21089784] 

8. Park HS, Lee SM, Kim HP, Seo JK, Chung YE. CT sinogram-consistency learning for metalinduced beam hardening correction. Med Phys. 2018;45(12):5376–5384. doi:10.1002/mp.13199 [PubMed: 30238586] 

9. Zhu Y, Zhao H, Wang T, et al. Sinogram domain metal artifact correction of CT via deep learning. Comput Biol Med. 2023;155:106710. doi:10.1016/j.compbiomed.2023.106710 [PubMed: 36842222] 

10. Peng C, Qiu B, Li M, et al. Gaussian diffusion sinogram inpainting for X-ray CT metal artifact reduction. Biomed Eng OnLine. 2017;16(1). doi:10.1186/s12938-016-0292-9 

11. Zhang H, Wang L, Li L, Cai A, Hu G, Yan B. Iterative metal artifact reduction for x-ray computed tomography using unmatched projector/backprojector pairs. Med Phys. 2016;43(6Part1):3019– 3033. doi:10.1118/1.4950722 [PubMed: 27277050] 

12. Zhang H, Dong B, Liu B. A reweighted joint spatial-radon domain CT image reconstruction model for metal artifact reduction. SIAM J Imaging Sci. 2018;11(1):707–733. doi:10.1137/17M1140212 

13. Chang Z, Ye DH, Srivastava S, Thibault JB, Sauer K, Bouman C. Prior-guided metal artifact reduction for iterative X-ray computed tomography. IEEE Trans Med Imaging. 2019;38(6):1532– 1542. doi:10.1109/TMI.2018.2886701 [PubMed: 30571617] 

14. De Man B, Nuyts J, Dupont P, Marchal G, Suetens P. An iterative maximumlikelihood polychromatic algorithm for CT. IEEE Trans Med Imaging. 2001;20(10):999–1008. doi:10.1109/42.959297 [PubMed: 11686446] 

15. Stayman JW, Otake Y, Prince JL, Khanna AJ, Siewerdsen JH. Model-based tomographic reconstruction of objects containing known components. IEEE Trans Med Imaging. 2012;31(10):1837–1848. doi:10.1109/TMI.2012.2199763 [PubMed: 22614574] 

16. Wang H, Li Y, He N, Ma K, Meng D, Zheng Y. DICDNet: deep interpretable convolutional dictionary network for metal artifact reduction in CT images. IEEE Trans Med Imaging. 2022;41(4):869–880. doi:10.1109/TMI.2021.3127074 [PubMed: 34752391] 

17. Lee J, Gu J, Ye JC. Unsupervised CT metal artifact learning using attention-guided β -cycleGAN. IEEE Trans Med Imaging. 2021;40(12):3932–3944. doi:10.1109/TMI.2021.3101363 [PubMed: 34329157] 

18. Wang J, Zhao Y, Noble JH, Dawant BM. Conditional generative adversarial networks for metal artifact reduction in CT images of the ear. In: Frangi AF, Schnabel JA, Davatzikos C, AlberolaLópez C, Fichtinger G, eds. Medical image computing and computer assisted intervention— MICCAI 2018. Vol 11070. Lecture Notes in Computer Science. Springer International Publishing; 2018:3–11. doi:10.1007/978-3-030-00928-1_1 

19. Nakao M, Imanishi K, Ueda N, Imai Y, Kirita T, Matsuda T. Regularized three-dimensional generative adversarial nets for unsupervised metal artifact reduction in head and neck CT images. IEEE Access. 2020;8:109453–109465. doi:10.1109/ACCESS.2020.3002090 

20. Liao H, Lin WA, Zhou SK, Luo J. ADN: artifact disentanglement network for unsupervised metal artifact reduction. IEEE Trans Med Imaging. 2020;39(3):634–643. doi:10.1109/ TMI.2019.2933425 [PubMed: 31395543] 

21. Zhang Y, Yu H. Convolutional neural network based metal artifact reduction in X-Ray computed tomography. IEEE Trans Med Imaging. 2018;37(6):1370–1381. doi:10.1109/TMI.2018.2823083 [PubMed: 29870366] 

22. Lin WA, Liao H, Peng C, et al. DuDoNet: dual domain network for CT metal artifact reduction. In: 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE; 2019:10504–10513. doi:10.1109/CVPR.2019.01076 

23. Peng C, Li B, Liang P, et al. A cross-domain metal trace restoring network for reducing X-Ray CT metal artifacts. IEEE Trans Med Imaging. 2020;39(12):3831–3842. doi:10.1109/ TMI.2020.3005432 [PubMed: 32746126] 

Med Phys. Author manuscript; available in PMC 2026 January 02. 

Haneda et al. 

Page 18 

24. Yu L, Zhang Z, Li X, Xing L. Deep sinogram completion with image prior for metal artifact reduction in CT images. IEEE Trans Med Imaging. 2021;40(1):228–238. doi:10.1109/ TMI.2020.3025064 [PubMed: 32956044] 

25. Yu L, Zhang Z, Li X, Ren H, Zhao W, Xing L. Metal artifact reduction in 2D CT images with selfsupervised cross-domain learning. Phys Med Biol. 2021;66(17):175003. doi:10.1088/1361-6560/ ac195c 

26. Xia J, Zhou Y, Deng W, et al. PND-Net: physics-inspired non-local dual-domain network for metal artifact reduction. IEEE Trans Med Imaging. 2024;43(6):2125–2136. doi:10.1109/ TMI.2024.3354925 [PubMed: 38236665] 

27. Gjesteby L, Shan H, Yang Q, et al. A dual-stream deep convolutional network for reducing metal streak artifacts in CT images. Phys Med Biol. 2019;64(23):235003. doi:10.1088/1361-6560/ ab4e3e [PubMed: 31618724] 

28. Wang T, Lu Z, Yang Z, et al. IDOL-Net: an interactive dual-domain parallel network for CT metal artifact reduction. IEEE Trans Radiat Plasma Med Sci. 2022;6(8):874–885. doi:10.1109/ TRPMS.2022.3171440 

29. Moon SG, Hong SH, Choi JY, et al. Metal artifact reduction by the alteration of technical factors in multidetector computed tomography: a 3-dimensional quantitative assessment. J Comput Assist Tomogr. 2008;32(4):630–633. doi:10.1097/RCT.0b013e3181568b27 [PubMed: 18664853] 

30. Stradiotti P, Curti A, Castellazzi G, Zerbi A. Metal-related artifacts in instrumented spine. Techniques for reducing artifacts in CT and MRI: state of the art. Eur Spine J. 2009;18(Suppl 1):102–108. doi:10.1007/s00586-009-0998-5 [PubMed: 19437043] 

31. Wellenberg RHH, Hakvoort ET, Slump CH, Boomsma MF, Maas M, Streekstra GJ. Metal artifact reduction techniques in muscu-loskeletal CT-imaging. Eur J Radiol. 2018;107:60–69. doi:10.1016/ j.ejrad.2018.08.010 [PubMed: 30292274] 

32. Selles M, Van Osch JAC, Maas M, Boomsma MF, Wellenberg RHH. Advances in metal artifact reduction in CT images: a review of traditional and novel metal artifact reduction techniques. Eur J Radiol. 2024;170:111276. doi:10.1016/j.ejrad.2023.111276 [PubMed: 38142571] 

33. Wu M, Keil A, Constantin D, Star-Lack J, Zhu L, Fahrig R. Metal artifact correction for x-ray computed tomography using kV and selective MV imaging. Med Phys. 2014;41(12):121910. doi:10.1118/1.4901551 [PubMed: 25471970] 

34. Huang JY, Kerns JR, Nute JL, et al. An evaluation of three commercially available metal artifact reduction methods for CT imaging. Phys Med Biol. 2015;60(3):1047–1067. doi:10.1088/0031-9155/60/3/1047 [PubMed: 25585685] 

35. Meyer E, Raupach R, Lell M, Schmidt B, Kachelrieß M. Frequency split metal artifact reduction (FSMAR) in computed tomography. Med Phys. 2012;39(4):1904–1916. doi:10.1118/1.3691902 [PubMed: 22482612] 

36. Lin WA, Liao H, Peng C, et al. DuDoNet: dual domain network for CT metal artifact reduction. Published online 2019. doi:10.48550/ARXIV.1907.00273 

37. AAPM CT Metal Artifact Reduction (CT-MAR) Grand Challenge. https://www.aapm.org/ GrandChallenge/CT-MAR/ 

38. AAPM CT Metal Artifact reduction (CT-MAR) grand challenge benchmark tool. GitHub. https:// github.com/xcist/example/tree/main/AAPM_datachallenge 

39. Grand challenges: fostering progress in medical physics through insights from leading teams and organizers. Presented at: AAPM 2024 66th annual meeting & exhibition; 07212024; Los Angels. 

40. Wu M, FitzGerald P, Zhang J, et al. XCIST—an open access x-ray/CT simulation toolkit. Phys Med Biol. 2022;67(19):194002. doi:10.1088/1361-6560/ac9174 

41. De Man B, Basu S, Chandra N. CatSim: a new computer assisted tomography simulation environment. In: Hsieh J, Flynn MJ, eds. Medical Imaging 2007: Physics of Medical Imaging. SPIE; 2007:65102G. doi:10.1117/12.710713 

42. Yan K, Wang X, Lu L, Summers RM. DeepLesion: automated mining of large-scale lesion annotations and universal lesion detection with deep learning. J Med Imaging. 2018;5(03):1. doi:10.1117/1.JMI.5.3.036501 

43. Goren N, Dowrick T, Avery J, Holder D, UCLH Stroke Eit Dataset—Radiology Data. Published online August 3, 2017. doi:10.5281/ZENODO.1199398 

Med Phys. Author manuscript; available in PMC 2026 January 02. 

Haneda et al. 

Page 19 

44. AAPM CT Metal Artifact Reduction (CT-MAR) Grand Challenge Scoring Metrics. Accessed July 7, 2025. https://github.com/xcist/example/blob/main/AAPM_datachallenge/scoring_metric.md 

45. Fan Y, Pack J, De Man B. A virtual imaging trial framework to study cardiac CT blooming artifacts. In: Stayman JW, ed. 7th International Conference on Image Formation in X-Ray Computed Tomography. SPIE; 2022:16. doi:10.1117/12.2646407 

46. Ohnesorge B, Flohr T, Klingenbeck-Regn K. Efficient object scatter correction algorithm for third and fourth generation CT scanners . Eur Radiol. 1999;9(3):563–569. doi:10.1007/s003300050710 [PubMed: 10087134] 

47. FitzGerald P, Araujo S, Wu M. parameterized spectrum estimation for x-ray computed tomography. Med Phys. 2021;48(5):2199–2213. doi:10.1002/mp.14715 [PubMed: 33426704] 

48. Zhang J, Wu M, FitzGerald P, Araujo S, De Man B. Development and tuning of models for accurate simulation of CT spatial resolution using CatSim. Phys Med Biol. 2024;69(4). doi:10.1088/1361-6560/ad2122 

49. Wang Z, Bovik AC, Sheikh HR, Simoncelli EP. Image quality assessment: from error visibility to structural similarity. IEEE Trans Image Process. 2004;13(4):600–612. doi:10.1109/ TIP.2003.819861 [PubMed: 15376593] 

50. Tao X, Wang Y, Lin L, Hong Z, Ma J. Learning to reconstruct CT images from the VVBP-tensor. IEEE Trans Med Imaging. 2021;40(11):3030–3041. doi:10.1109/TMI.2021.3090257 [PubMed: 34138703] 

51. He J, Wang Y, Ma J. Radon inversion via deep learning. IEEE Trans Med Imaging. 2020;39(6):2076–2087. doi:10.1109/TMI.2020.2964266 [PubMed: 31944948] 

52. Guo Y, Wang Y, Zhu M, et al. Dual domain closed-loop learning for sparse-view CT reconstruction. In: Stayman JW, ed. 7th International Conference on Image Formation in X-Ray Computed Tomography. SPIE; 2022:60. doi:10.1117/12.2646639 

53. Wu Q, Chen L, Wang C, et al. Unsupervised polychromatic neural representation for CT metal artifact reduction. In: Proceedings of the 37th International Conference on Neural Information Processing Systems. NIPS ‘23. Curran Associates Inc.; 2023:69605–69624. 

54. Wu Q, Guo X, Chen L, et al. Unsupervised density neural representation for CT metal artifact reduction. CoRR. 2024:abs/2405.07047. doi:10.48550/ARXIV.2405.07047 

55. Park HS, Seo JK, Jeon K. Implicit neural representation-based method for metal-induced beam hardening artifact reduction in X-ray CT imaging. Med Phys. 2025;52(4):2201–2211.doi:10.1002/ mp.17649 [PubMed: 39888006] 

56. Sitzmann V, Martel JNP, Bergman AW, Lindell DB, Wetzstein G, Implicit neural representations with periodic activation functions. In: Proceedings of the 34th International Conference on Neural Information Processing Systems. NIPS ‘20. Curran Associates Inc.; 2020:7462–7473. 

57. Nichol A, Achiam J, Schulman J. On first-order meta-learning algorithms. arXivorg. 2018. Accessed April 15, 2025. https://arxiv.org/abs/1803.02999v3 

58. Park HS, Jeong YJ, Jeon K. A Robust multidomain network for short-scanning amyloid PET image restoration. IEEE Trans Radiat Plasma Med Sci. 2025;9(1):57–68. doi:10.1109/ TRPMS.2024.3430298 

59. Hatamizadeh A, Nath V, Tang Y, Yang D, Roth HR, Xu D. Swin UNETR: swin transformers for semantic segmentation of brain tumors in MRI images. In: Crimi A, Bakas S, eds. Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries. Springer International Publishing; 2022:272–284. doi:10.1007/978-3-031-08999-2_22 

60. Adler J, Kohr H, Öktem O, Operator discretization library (ODL). Published online January 17, 2017. doi:10.5281/zenodo.249479 

61. Fan F, Ritschl L, Beister M, et al. Simulation-driven training of vision transformers enables metal artifact reduction of highly truncated CBCT scans. Med Phys. 2024;51(5):3360–3375. doi:10.1002/mp.16919 [PubMed: 38150576] 

62. Ronneberger O, Fischer P, Brox T. U-Net: convolutional networks for biomedical image segmentation. In: Navab N, Hornegger J, Wells WM, Frangi AF, eds. Medical image computing and computer-assisted intervention—MICCAI 2015. Vol 9351. Lecture Notes in Computer Science. Springer International Publishing:234–241. doi:10.1007/978-3-319-24574-4_28 

Med Phys. Author manuscript; available in PMC 2026 January 02. 

Haneda et al. 

Page 20 

63. He K, Zhang X, Ren S, Sun J, Deep residual learning for image recognition. Published online December 10, 2015. doi:10.48550/arXiv.1512.03385 

64. Goodfellow I, Pouget-Abadie J, Mirza M, et al. Generative adversarial networks. Commun ACM. 2020;63(11):139–144. doi:10.1145/3422622 

65. Ho J, Jain A, Abbeel P. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems. Curran Associates, Inc.; 2020:6840–6851. Accessed April 11, 2025. https:// proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html 

66. Song Y, Sohl-Dickstein J, Kingma DP, Kumar A, Ermon S, Poole B, Score-based generative modeling through stochastic differential equations. Published online February 10, 2021. doi:10.48550/arXiv.2011.13456 

67. Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need. Advances in Neural Information Processing Systems. Curran Associates, Inc.; 2017. Accessed April 11, 2025. https:// papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html 

68. Cao H, Wang Y, Chen J, et al. Swin-Unet: unet-like pure transformer for medical image segmentation. Published online 2021. doi:10.48550/ARXIV.2105.05537 

69. Zamir SW, Arora A, Khan S, Hayat M, Khan FS, Yang MH, Restormer: efficient transformer for high-resolution image restoration. In: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE; 2022:5718–5729. doi:10.1109/CVPR52688.2022.00564 

70. Liu GH, Vahdat A, Huang DA, Theodorou EA, Nie W, Anandkumar A, I2SB: image-to-image Schrödinger bridge. Published online May 26, 2023. doi:10.48550/arXiv.2302.05872 

71. Wang Z, Jiang Y, Zheng H, et al. Patch diffusion: faster and more data-efficient training of diffusion models. In: Proceedings of the 37th International Conference on Neural Information Processing Systems. NIPS ‘23. Curran Associates Inc.; 2023:72137–72154. 

72. Wang Y, Yu J, Zhang J, Zero-shot image restoration using denoising diffusion null-space model. Published online December 7, 2022. doi:10.48550/arXiv.2212.00490 

73. Wang T, Xia W, Huang Y, et al. DAN-Net: dual-domain adaptive-scaling non-local network for CT metal artifact reduction. Phys Med Biol. 2021;66(15). doi:10.1088/1361-6560/ac1156 

74. Zeng GL. A projection-domain iterative algorithm for metal artifact reduction by minimizing the total-variation norm and the negativepixel energy. Vis Comput Ind Biomed Art. 2022;5(1). doi:10.1186/s42492-021-00094-w 

75. Suvorov R, Logacheva E, Mashikhin A, et al. Resolution-robust large mask inpainting with fourier convolutions. In: 2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). 2022:3172–3182. doi:10.1109/WACV51458.2022.00323 

76. Chi L, Jiang B, Mu Y. Fast fourier convolution. Advances in Neural Information Processing Systems. Curran Associates, Inc.; 2020:4479–4488. Accessed April 15, 2025. https:// papers.nips.cc/paper/2020/hash/2fd5d41ec6cfab47e32164d5624269b1-Abstract.html 

77. Isola P, Zhu JY, Zhou T, Efros AA, Image-to-image translation with conditional adversarial networks. In: 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE; 2017:5967–5976. doi:10.1109/CVPR.2017.632 

78. Geirhos R, Rubisch P, Michaelis C, Bethge M, Wichmann FA, Brendel W, ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. In: 7th International Conference on Learning Representations. OpenReview.net; 2019. 

79. Niwa S, Ichikawa K, Kawashima H, Takata T, Minami S, Mitsui W. Reduction of streak artifacts caused by low photon counts utilizing an image-based forward projection in computed tomography. Comput Biol Med. 2021;135:104583. doi:10.1016/j.compbiomed.2021.104583 [PubMed: 34216891] 

80. Morioka Y, Ichikawa K, Kawashima H. Quality improvement of images with metal artifact reduction using a noise recovery technique in computed tomography. Phys Eng Sci Med. 2024;47(1):169–180. doi:10.1007/s13246-023-01353-1 [PubMed: 37938518] 

81. Ichikawa K, Kawashima H, Takata T. An image-based metal artifact reduction technique utilizing forward projection in computed tomography. Radiol Phys Technol. 2024;17(2):402–411. doi:10.1007/s12194-024-00790-1 [PubMed: 38546970] 

82. Ichikawa K, Kawashima H, Shimada M, Adachi T, Takata T. A three-dimensional crossdirectional bilateral filter for edgepreserving noise reduction of low-dose computed tomography 

Med Phys. Author manuscript; available in PMC 2026 January 02. 

Haneda et al. 

Page 21 

images. Comput Biol Med. 2019;111:103353. doi:10.1016/j.compbiomed.2019.103353 [PubMed: 31306807] 

83. Karras T, Aittala M, Laine S, et al. Alias-free generative adversarial networks. In: Proceedings of the 35th International Conference on Neural Information Processing Systems. NIPS ‘21. Curran Associates Inc.; 2021:852–863. 

84. Heusel M, Ramsauer H, Unterthiner T, Nessler B, Hochreiter S, GANs trained by a two time-scale update rule converge to a local nash equilibrium. In: Proceedings of the 31st International Conference on Neural Information Processing Systems. NIPS’17. Curran Associates Inc.; 2017:6629–6640. 

85. Karras T, Laine S, Aittala M, Hellsten J, Lehtinen J, Aila T, Analyzing and Improving the image quality of StyleGAN. In: 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2020:8107–8116. doi:10.1109/CVPR42600.2020.00813 

86. Mescheder L, Geiger A, Nowozin S, Which Training Methods for GANs do actually Converge? In: PMLR; 2018:3481–3490. Accessed April 15, 2025. https://hdl.handle.net/21.11116/0000-000ADFD4-C 

87. Özbey M, Dalmaz O, Dar SUH, et al. Unsupervised medical image translation with adversarial diffusion models. IEEE Trans Med Imaging. 2023;42(12):3524–3539. doi:10.1109/ TMI.2023.3290149 [PubMed: 37379177] 

88. Jeevan P, Srinidhi A, Prathiba P, WaveMixSR SethiA, : Resource-efficient neural network for image super-resolution. In: 2024 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE; 2024:5872–5880. doi:10.1109/WACV57701.2024.00578 

89. Lehtinen J, Munkberg J, Hasselgren J, et al. Noise2Noise: learning image restoration without clean data. arXivorg. 2018. Accessed April 15, 2025. https://arxiv.org/abs/1803.04189v3 

Med Phys. Author manuscript; available in PMC 2026 January 02. 

Haneda et al. 

Page 22 

## **FIGURE 1.** 

AAPM CT-MAR grand challenge structure. The challenge consisted of three phases. In phase 1, training datasets were provided to the participants for their MAR algorithm development. In phase 2, five clinical datasets were provided to the participants five for the preliminary scoring. In phase 3, 29 clinical datasets were provided, and the final scores and ranking were computed using the scoring metrics. 

Med Phys. Author manuscript; available in PMC 2026 January 02. 

Haneda et al. 

Page 23 

## **FIGURE 2.** 

Types of images and sinograms that were provided during the challenge. (1)–(5) were provided for training phase. Only (2) and (4) were provided for feedback phase and final scoring phase. Participants were asked to submit an image without any metal artifacts but that still contains metal objects as shown in (6). 

Med Phys. Author manuscript; available in PMC 2026 January 02. 

Haneda et al. 

Page 24 

## **FIGURE 3.** 

Examples of training datasets are shown. From left to right, five types of data in each training dataset are displayed: two sinograms (without and with metal objects), two reconstructed images (without and with metal objects), and a metal mask. The top row is an abdominal region with three stainless steel objects and the bottom row is a head region with three amalgam objects. The display window for the reconstructed images is W/L = 1000/0 HU. The display window for the sinograms is [min max] = [0 8] in p value. 

Med Phys. Author manuscript; available in PMC 2026 January 02. 

Page 25 

Haneda et al. 

## **FIGURE 4.** 

The 1[st] place MAR algorithm overview. PSC: photon starvation correction, BHC: beam hardening correction, NC: noise correction, SC: scatter correction, BHM: beam hardening modeling, NM: noise modeling, SM: scatter modeling, PSM: photon starvation modeling. In the primal mapping, FBP algorithm reconstructs an image between BHC and NC. In the dual mapping, reprojection (PROJ) is performed on the estimated density map, and multienergy projection data are calculated by incorporating X-ray spectrum and material-specific attenuation profiles, thereby enabling beam hardening modeling. 

Med Phys. Author manuscript; available in PMC 2026 January 02. 

Page 26 

Haneda et al. 

**FIGURE 5.** 

The 2[nd] place MAR algorithm: (a) Schematic diagram of the MAR approach. (b) Network architecture for the residual artifact correction. 

Med Phys. Author manuscript; available in PMC 2026 January 02. 

Haneda et al. 

Page 27 

**FIGURE 6.** 

The 3[rd] place MAR algorithm: the 30 cross-domain framework for MAR, which includes of segmentation network, linear interpolation, DICDNet and inpainting network. 

Med Phys. Author manuscript; available in PMC 2026 January 02. 

Haneda et al. 

Page 28 

## **FIGURE 7.** 

Statistics of deep learning network architecture used by the challenge participants. 92% of the participants used deep learning and 8% used analytical approaches (non-DL). We categorized network architectures into UNet, ResNet, GAN, diffusion models, transformers, and others. Most of the teams used more than one network architecture to implement MAR. 

Med Phys. Author manuscript; available in PMC 2026 January 02. 

Haneda et al. 

Page 29 

## **FIGURE 8.** 

Scoring metric distributions from all 26 participants. Each score metric was normalized to 0.0–4.0 (0: good, 4: bad) and averaged across the 29 scoring datasets. The “total” score is an average across the eight metrics for all cases. 

Med Phys. Author manuscript; available in PMC 2026 January 02. 

Haneda et al. 

Page 30 

## **FIGURE 9.** 

RMSE, SSIM, and PSNR results on 1,000 test images that are independent from scoring datasets by 23 participants. Three participants were excluded from the plot due to the large bias and variance for displaying purpose. The MAR processed images were clipped by the range of [−2000 6000] HU and normalized before calculation. 

Med Phys. Author manuscript; available in PMC 2026 January 02. 

Haneda et al. 

Page 31 

## **FIGURE 10.** 

MAR processed images by selected teams (1[st] , 2[nd] , 3[rd] , 10[th] , and 21[st] place) in the final ranking along with NMAR, uncorrected, and ground truth (GT) images for head with dental fillings (top), chest and spine with screws (middle), and pelvis with a prosthesis (bottom). The ranking for each specific image could be different from the final ranking. The display window was set to [min max] = [−100, 100] HU to observe the tissue regions contaminated by metal artifacts. 

Med Phys. Author manuscript; available in PMC 2026 January 02. 

Haneda et al. 

Page 32 

## **FIGURE 11.** 

ROI placements to compute evaluation metrics for three cases: head with dental fillings (left), thorax with spinal screws (middle), and pelvis with a prosthesis (right). 

Med Phys. Author manuscript; available in PMC 2026 January 02. 

Haneda et al. 

Page 33 

|**Ranking**<br>**Final score**<br>**Algorithm domain**<br>**Provided training data used?**<br>**Metrics used for optimization**<br>**Deep learning types**|1<br>0.96<br>Sino, image<br>Yes, and added more<br>CT number<br>CNN, UNet52<br>2<br>0.98<br>Sino, image<br>Yes<br>CT number, metal integ<br>Implicit neural representation, ResNet7,53–55<br>3<br>0.99<br>Sino, image<br>Yes<br>CT number, metal integ<br>Swin Transformer-based UNet, ResNet46,47<br>4<br>1.05<br>Image<br>Yes<br>CT number, sharpness, SSIM<br>U-shape Transformer<br>5<br>1.06<br>Sino, image<br>Yes<br>CT number<br>Latent Diffusion model, Attention UNet<br>6<br>1.09<br>Image<br>Unknown<br>Unknown<br>Diffusion model<br>7<br>1.15<br>Image<br>No<br>CT number, SSIM, bone integ, metal integ<br>Transformer based network: Restormer, Swin-UNet68,69<br>8<br>1.16<br>Image<br>Yes<br>CT number, sharpness, SSIM<br>Diffusion model for image-to-image translation (I2SB)70,71<br>9<br>1.21<br>Sino, image<br>Yes<br>CT number, metal integ, streak<br>Diffusion model21<br>10<br>1.39<br>Sino, image<br>Yes<br>CT number, sharpness<br>GAN family with Transformer based Model<br>11<br>1.42<br>Image<br>Yes<br>All<br>GAN family<br>12<br>1.45<br>Sino, image<br>Yes<br>CT number<br>ResUNet and Diffusion model72<br>13<br>1.50<br>Sino, image<br>Unknown<br>Unknown<br>Physics based non-local dual-domain network26,73<br>14<br>1.52<br>Sino<br>No<br>CT number, noise, metal integ, streak<br>GAN based LaMa (Fast Fourier Convolution)<br>15<br>1.52<br>Sino<br>No<br>TV norm<br>Non-DL74<br>16<br>1.58<br>Sino, image<br>Yes<br>CT number<br>Diffusion model, UNet<br>17<br>1.63<br>Sino, image<br>Yes, and added more<br>Others<br>GAN based LaMa (Fast Fourier Convolution)75–78<br>18<br>1.68<br>Sino<br>Yes<br>CT number<br>Diffusion network, UNet<br>19<br>1.73<br>Sino<br>Yes<br>CT number, noise, streak<br>Non-DL (NMAR family)7,79–82<br>20<br>1.93<br>Image<br>Yes<br>CT number metal integ<br>UNet<br>21<br>2.05<br>Sino, image<br>Yes<br>SSIM, metal integ, streak<br>UNet (ResNet34), TransUNet<br>22<br>2.09<br>Image<br>Yes<br>CT number<br>UNet<br>23<br>2.32<br>Image<br>Yes<br>Noise, SSIM<br>StyleGAN383–86<br>24<br>2.53<br>Sino<br>Yes<br>CT number, sharpness, SSIM, metal integ<br>UNet, ResNet<br>25<br>2.86<br>Sino<br>Yes<br>CT number, noise, SSIM<br>Adversarial diffusion model, UNet, WaveMix87,88<br>26<br>3.34<br>Sino<br>Yes<br>CT number, noise<br>UNet, Noise2Noise89<br>NMAR<br>1.82<br>Sino<br>N/A<br>N/A<br>N/A|
|---|---|



Med Phys. Author manuscript; available in PMC 2026 January 02. 

Haneda et al. 

Page 34 

|**Ranking**<br>**Final score**<br>**CT number**<br>**Noise**<br>**Sharpness**<br>**Streak**<br>**SSIM**<br>**Metal integrity**<br>**Bone integrity**<br>**PBR**|1<br>0.96<br>**0.74**<br>**0.01**<br>0.72<br>**1.29**<br>**0.75**<br>1.46<br>**0.89**<br>1.79<br>2<br>0.98<br>0.81<br>0.19<br>**0.52**<br>1.65<br>0.95<br>1.09<br>0.99<br>**1.62**<br>3<br>0.99<br>0.82<br>**0.01**<br>0.54<br>1.48<br>0.98<br>**0.96**<br>1.09<br>2.03<br>10<br>1.39<br>1.15<br>0.44<br>0.64<br>2.15<br>1.44<br>1.33<br>1.42<br>2.54<br>21<br>2.04<br>1.72<br>0.19<br>1.21<br>1.90<br>3.57<br>1.24<br>2.65<br>3.86<br>NMAR<br>1.82<br>2.03<br>1.06<br>0.58<br>1.99<br>2.03<br>2.06<br>2.05<br>2.76|
|---|---|



Med Phys. Author manuscript; available in PMC 2026 January 02. 

