# Abadi et al. — AAPM Truth-based CT (TrueCT) Reconstruction Grand Challenge (Med. Phys. 2025)

_Source: <https://pmc.ncbi.nlm.nih.gov/articles/PMC11973969/>_

_PDF: `papers/abadi_2025_truect.pdf`_

---

## **HHS Public Access** 

## Author manuscript 

Med Phys. Author manuscript; available in PMC 2025 April 07. 

Published in final edited form as: 

Med Phys. 2025 April ; 52(4): 1978–1990. doi:10.1002/mp.17619. 

## **AAPM Truth-based CT (TrueCT) reconstruction grand challenge** 

**Ehsan Abadi**[1,2,3] , **W. Paul Segars**[1,2,4] , **Nicholas Felice**[1,2] , **Saman Sotoudeh-Paima**[1,3] , **Eric A. Hoffman**[5] , **Xiao Wang**[6] , **Wei Wang**[7] , **Darin Clark**[1,8] , **Siqi Ye**[9] , **Giavanna Jadick**[10] , **Milo Fryling**[1] , **Donald P. Frush**[1] , **Ehsan Samei**[1,2,3,4,11] 

1Center for Virtual Imaging Trial, Carl E. Ravin Advanced Imaging Laboratories, Department of Radiology, Duke University School of Medicine, Durham, North Carolina, USA 

2Medical Physics Graduate Program, Duke University, Durham, North Carolina, USA 

3Department of Electrical & Computer Engineering, Duke University, Durham, North Carolina, USA 

4Department of Biomedical Engineering, Duke University, Durham, North Carolina, USA 

5Department of Radiology, Internal Medicine and Biomedical Engineering, University of Iowa, Iowa City, Iowa, USA 

6Computational Science and Engineering Division, Oak Ridge National Laboratories, Oak Ridge, Tennessee, USA 

7Institute of Applied Mathematics, Shenzhen Polytechnic, Shenzhen, Guangdong, China 

8Quantitative Imaging and Analysis Lab, Department of Radiology, Duke University, Durham, North Carolina, USA 

9Department of Radiation Oncology, Stanford University, Stanford, California, USA 

10Department of Radiology, University of Chicago, Chicago, Illinois, USA 

11Department of Physics, Duke University, Durham, North Carolina, USA 

## **Abstract** 

**Background:** This Special Report summarizes the 2022, AAPM grand challenge on Truth-based CT image reconstruction. 

**Purpose:** To provide an objective framework for evaluating CT reconstruction methods using virtual imaging resources consisting of a library of simulated CT projection images of a population of human models with various diseases. 

**Methods:** Two hundred unique anthropomorphic, computational models were created with varied diseases consisting of 67 emphysema, 67 lung lesions, and 66 liver lesions. The organs were 

> **Correspondence** Ehsan Abadi, Center for Virtual Imaging Trial, Carl E. Ravin Advanced Imaging Laboratories, Department of Radiology, Duke University School of Medicine, Durham, NC 27705, USA. ehsan.abadi@duke.edu. CONFLICT OF INTEREST STATEMENT 

> Unrelated to this study, Ehsan Abadi has relationship with Siemens, GE, and Silomedics, LLC. W. Paul Segars has relationship with Silomedics, LLC. Ehsan Samei has relationships with GE, Siemens, Imalogix, 12Sigma, Sun-Nuclear, Metis Health Analytics, Silomedics, Cambridge University Press, and Wiley and Sons. 

Abadi et al. 

Page 2 

modeled based on clinical CT images of real patients. The emphysematous regions were modeled using segmentations from patient CT cases in the COPDGene Phase I dataset. For the lung and liver lesion cases, 1–6 malignant lesions were created and inserted into the human models, with lesion diameters ranging from 5.6 to 21.9 mm for lung lesions and 3.9 to 14.9 mm for liver lesions. The contrast defined between the liver lesions and liver parenchyma was 82 ± 12 HU, ranging from 50 to 110 HU. Similarly, the contrast between the lung lesions and the lung parenchyma was defined as 781 ± 11 HU, ranging from 725 to 805 HU. For the emphysematous regions, the defined HU values were −950 ± 17 HU ranging from −918 to −979 HU. The developed human models were imaged with a validated CT simulator. The resulting CT sinograms were shared with the participants. The participants reconstructed CT images from the sinograms and sent back their reconstructed images. The reconstructed images were then scored by comparing the results against the corresponding ground truth values. The scores included both task-generic (root mean square error [RMSE] and structural similarity matrix [SSIM]), and task-specific (detectability index [d’] and lesion volume accuracy) metrics. For the cases with multiple lesions, the measured metric was averaged across all the lesions. To combine the metrics with each other, each metric was normalized to a range of 0 to 1 per disease type, with “0” and “1” being the worst and best measured values across all cases of the disease type for all received reconstructions. 

**Results:** The True-CT challenge attracted 52 participants, out of which 5 successfully completed the challenge and submitted the requested 200 reconstructions. Across all participants and disease types, SSIM absolute values ranged from 0.22 to 0.90, RMSE from 77.6 to 490.5 HU, d’ from 0.1 to 64.6, and volume accuracy ranged from 1.2 to 753.1 mm[3] . The overall scores demonstrated that participant “A”had the best performance in all categories, except for the metrics of d’ for lung lesions and RMSE for liver lesions. Participant “A” had an average normalized score of 0.41 ± 0.22, 0.48 ± 0.32, and 0.42 ± 0.33 for the emphysema, lung lesion, and liver lesion cases, respectively. 

**Conclusions:** The True-CT challenge successfully enabled objective assessment of CT reconstructions with the unique advantage of access to a diverse population of diseased human models with known ground truth. This study highlights the significant potential of virtual imaging trials in objective assessment of medical imaging technologies. 

## **Keywords** 

AAPM grand challenge; computational phantoms; computed tomography; CT reconstruction; imaging simulators; in silico trials; medical imaging simulations; virtual imaging trials 

## **1 | INTRODUCTION** 

Image reconstruction is an essential process in CT imaging, transforming acquired projection data into volumetric images. Initially, CT reconstruction techniques relied solely on filtered back-projection (FBP) algorithms, which have several limitations due to their reliance on simple mathematical assumptions and linear models, leading to compromised image quality particularly at lower radiation doses.[1] More recent, advanced image reconstruction methods provide higher image quality and thereby improve clinical diagnosis and disease assessments.[2] Two notable examples are iterative reconstruction and deep learning methods that have been increasingly developed and utilized with promising 

Med Phys. Author manuscript; available in PMC 2025 April 07. 

Abadi et al. 

Page 3 

performances.[1–6] These advanced reconstruction techniques are often data-driven and/or based on approximations of their own which can distort relationship between sinogram measurements and reconstructed images. As such, there is a need for resources that can objectively evaluate CT reconstruction algorithms across various clinical conditions and imaging tasks. 

Highest objectivity in the assessment of reconstruction algorithms requires reference datasets of projection images from objects whose ground truth is precisely known and further represent diverse anatomical and disease realities. Neither clinical patient images nor physical phantoms meet these requirements, as clinical patient images lack the full knowledge of the exact patient anatomy and pathophysiology, and physical phantoms do not provide anthropomorphic realism. Virtual imaging tools[7] provide an alternative method for assessing CT reconstructions. By deploying computational models of humans and imaging systems, virtual imaging tools offer both realism and ground truth needed for a relevant and objective assessment of CT reconstruction techniques. 

This study aimed to utilize a virtual imaging framework to build a library of simulated CT projection images of a population of human models with various diseases. The generated library was used in an AAPM grand challenge, characterized as Truth-based CT reconstruction (True-CT),in which participants were invited to try reconstructing CT images from simulated sinograms acquired from the human models. The ground truth that was not made available to participants was used to objectively evaluate their outcomes. The paper reports the findings of this grand challenge. 

## **2 | METHODS** 

The dataset was created for the True-CT challenge, ordained in 2022 by the AAPM grand challenges committee and led by the Center for Virtual Imaging Trials.[8] The challenge framework is illustrated in Figure 1. In summary, 200 unique anthropomorphic, computational phantoms were created. These human models were imaged with a validated CT simulator. The resulting CT sinograms were shared with the challenge participants. The participants reconstructed CT images from the acquired sinograms and sent back their reconstructed images. The reconstructed images were then scored by comparing them against, their corresponding ground truths. The following describes each step in details. 

## **2.1 | Human models** 

A total of 200 anatomically variable, human models were developed with varied disease conditions (i.e., 67 cases with emphysema, 67 with lung lesions, and 66 with liver lesions). The body habitus of these models were created from CT images of 200 unique patients. 

The human models with liver lesions (N = 66) were created based on the original library of extended cardiac-torso (XCAT) models.[9] Sixty-six (29 females) XCATs were selected representing a diverse range of body habitus. These human models were created based on patient CT cases. The patients had an average age of 46 ± 18 years, ranging from 15 to 78 years. Their average body mass index was 26.9 ± 5.4 kg/m[2] , ranging from 15.5 to 38.8 kg/m[2] . All the liver models incorporated hepatic vasculature using a physics-based vessel 

Med Phys. Author manuscript; available in PMC 2025 April 07. 

Abadi et al. 

Page 4 

generation algorithm[10] and liver lesions. A total of 246 liver lesions were modeled based on a technique described earlier, modeling malignant lesions.[11] Following consultations with a radiologist, seven of the human models (~10% of the cases) were set to contain only one lesion. The one lesion paradigm is particularly important to model because they represent a single abnormality with a significant difference in management options compared to patients with multiple lesions. The remaining human models included 2–6 lesions with the number being randomly selected. Lesion sizes were selected to reflect clinical variability, with diameters varying from 3.9 to 14.9 mm with an average of 8.5 mm. The lesions were placed randomly within the liver parenchyma of each phantom. 

To develop the human models with emphysema,[12] 67 (32 females) patient CT cases were randomly chosen from the COPDGene Phase I dataset. The patients had an average age of 62 ± 9 years, ranging from 45 to 79 years. Their average body mass index was 28.7 ± 6.0 kg/m[2] ,ranging from 18.71 to 41.7 kg/m[2] .The major organs (e.g., body, bones, lungs, heart, liver) were segmented using commercial software.[13–15] The airways, pulmonary vessels, and emphysematous regions were segmented using an algorithm provided by the University of Iowa.[16] The organ and intra-organ segmentation masks were fit with mesh surfaces using the marching cubes algorithm from the Visualization Toolkit (VTK).[17] The meshes were imported into the Rhinoceros 3D modeling software (McNeel, Seattle, USA), then smoothed and refined to avoid unrealistic sharp edges due to the spatial resolution and segmentation limitations.This was done by manually smoothing the meshes using the “Smooth” function within Rhinoceros then using the “QuadRemesh”function to redefine each mesh with an optimized topology. 

A similar approach was used to model lung lesions. The body habitus was modeled based on 67 (40 females) patient CT cases selected randomly from the COPDGene Phase I datasets, separate from the cases that were used for emphysema. The patients had an average age of 61 ± 9 years,ranging from 47 to 80 years. Their average body mass index was 28.1 ± 7.0 kg/m[2] , ranging from 13.0 to 50.5 kg/m[2] . Each CT case was segmented and fit with optimized mesh surfaces using the methods described above. A total of 204 lung lesions were created using the techniques described earlier, modeling malignant lesions.[18] Per consultation with a radiologist, 22 of the human models (~30% of the cases) included only one lesion. The one lesion paradigm had a higher percentage in the lung lesions compared to the liver lesions, because they are more prevalent.[19,20] The remaining cases were randomly chosen to have 2–6 lesions. The size of each lesion was randomly determined with the diameters ranging from 5.6 to 21.9 mm and an average of 11.4 mm.The lesions were randomly within the right and left lungs. 

All 200 human models were voxelized at an isotropic voxel size of 0.25 mm. Trabecular bone texture and lung parenchyma were incorporated using anatomically informed statistical models.[21–23] The material of each organ and structure was defined by assigning density and elemental compositions with values derived from the International Commission for Radiation Units (ICRU) report 46.[24] To enable portal venous phase contrast-enhanced acquisitions for the human models with liver lesions, the liver parenchyma, lesion, and vasculature materials were adjusted to include iodine with densities that represent a portal venous phase. For all materials, patient-to-patient variability was accounted for by varying 

Med Phys. Author manuscript; available in PMC 2025 April 07. 

Page 5 

Abadi et al. 

the average density values (per ICRU report) in the range of ± 15 HU. Specifically, the contrast defined between the liver lesions and liver parenchyma was 82 ± 12 HU, ranging from 50 to 110 HU. Similarly, the contrast between the lung lesions and the lung parenchyma was defined as 781 ± 11 HU, ranging from 725 to 805 HU. For the emphysematous regions, the defined HU values were −950 ± 17 HU ranging from −918 to −979 HU. 

## **2.2 | CT simulations** 

The developed human models were imaged using a CT simulator, DukeSim.[25,26] DukeSim generates projection images of voxelized computational models with scanner-specific or scanner-generic configurations. It has a ray-tracing module for calculating the primary images and a Monte Carlo module[27,28] for estimating the photon scattering process. DukeSim accounts for various features that clinical scanners have, including focal spot wobbling, tube current modulation,[29] anti-scatter grid, and beam hardening correction. The simulator has been validated against, experimental measurements of physical phantoms acquired from various scanners.[25–27,29–31] 

For the True-CT challenge, a generic scanner model (called Duke-1 model) was designed to have properties close to those of current modern CT scanners but representing no specific make or model. Duke-1 had an energy-integrating detector model with a cascaded physical process that accounts for the intrinsic x-ray properties of the scintillator material Gd2O2S with a thickness of 0.8 mm and a density of 7.8 g/cm[3] . The energy response for this detector model was estimated following established techniques[32,33] by calculating the x-ray deposition in the scintillator and the optical signal generation with a gain of 50 photons per keV x-ray. For the x-ray spectrum, a validated open-source software was used.[34] The geometrical definitions of Duke-1 are summarized in Table 1. 

Using the Duke-1 scanner model, the 200 developed human models were scanned following standard clinical imaging protocols. All scans were done helically with a pitch (i.e., gantry distance traveled in one rotation[mm] divided by beam collimation[mm]) of 1.0.The x-ray spectrum was set to 120 kV with an angle-dependent bowtie filter profile that was measured in a prior study.[35] Tube current modulation was incorporated following a previously validated technique.[29] To represent radiation dose variabilities observed in clinical scans, the virtual acquisitions were performed at varying dose levels ranging from 50 to 150 mAs with 25 mAs increments. 

The simulations output was 200 CT sinograms formatted in the DICOM-CT-PD format,[36 ] following the established work of the AAPM Low Dose CT grand challenge.[37] The DICOM-CT-PD is a DICOM-based file format designed to contain CT sinograms along with their corresponding acquisition parameters, which are essential for reconstructions. Some examples of these parameters include acquisition geometry (distances between source, isocenter, and detectors), focal spot positions per projection, number of projections per rotation, detector shapes and sizes, water attenuation coefficient, and tube current per projection. 

Med Phys. Author manuscript; available in PMC 2025 April 07. 

Page 6 

Abadi et al. 

## **2.3 | Data verifications** 

To ensure the credibility of the generated sinogram dataset, the developed human models and the CT simulations were both verified. The developed human models were verified in two steps. Initially, an imaging scientist reviewed all segmentation masks, identifying and manually correcting improperly-segmented organs or structures using the Rhinoceros 3D modeling software (McNeel, Seattle, USA). Once the models were fully developed and voxelized, a second imaging scientist conducted a quality control check. This involved examining all human models with the ImageJ viewer,[38] ensuring the absence of artifacts or non-anthropomorphic objects. The imaging scientist also verified that each human model contained the expected number of lesions and checked the material definitions to confirm the correct material assignment per structure. Any identified issues were addressed, and the problematic models were regenerated. 

Although the CT simulator used in this study has been validated in previous works,[25–] 27,29–31 further verifications were conducted to ensure its accuracy using the “Duke-1” system definition (Table 1). This involved test simulations of a water phantom and an ACR phantom. The resulting CT sinograms were reconstructed using an open-source CT reconstruction toolkit (MCR Toolkit).[39] The reconstructed images were visually inspected to confirm the absence of potential artifacts. Additionally, CT numbers were verified in the water phantom and in the inserts of the ACR phantom. Further, the CT sinograms generated from the 200 human models were reconstructed with the MCR toolkit and the results inspected by an imaging scientist to ensure they were free from artifacts and nonanthropomorphic structures. 

## **2.4 | Data dissemination** 

The imaging community at large was invited to take part in the challenge through a registration process managed by the AAPM.T he generated CT sinograms were shared with the participants of the challenge using the Medical Image Challenges Initiative (MedICI) platform.[40] In addition to the CT sinograms of the 200 human models,a CT sinogram and a sample reconstruction of a test object (a cylindrical phantom with multiple inserts) were provided to assist the participants in preparing and testing their codes, allowing them a method to verify the correct operation of their reconstruction algorithm. 

The participants were given a document introducing the challenge and providing an overview of the dataset, detailing the required settings and imaging format for the reconstructed images (Table 2). The participants were asked to provide full reconstruction of the “z” field of view for each sinogram data they received. These standardized settings enabled consistent evaluation of the received reconstructions. The document also clarified the methods for scoring the images. 

## **2.5 | Scoring** 

The received reconstructed images were scored in terms of two task-generic image quality metrics: structural similarity index (SSIM) and root mean square error (RMSE). These metrics were measured between the pairs of the reconstructed images and their corresponding digital ground truth. The digital ground truth was defined as the mono- 

Med Phys. Author manuscript; available in PMC 2025 April 07. 

Abadi et al. 

Page 7 

energetic representation of the human models in Hounsfield Units (HU) at the effective energy (i.e., 64 keV) of the acquisitions. To enable voxel-wise assessment,CT images were registered (i.e., upscaled) to the voxel size of the corresponding human models (0.25 mm isotropic voxel size) using a linear interpolation technique. 

Both SSIM and RMSE were measured in volumes of interest (VOI) per disease type. For the emphysema cases, the measurements were done in the lung parenchyma regions. For the liver and lung lesion cases, the VOIs were defined as twice the size of the corresponding lesions. The VOIs were specified by dilating the mask of each lesion, known from the ground truth data, with a spherical structural element. The SSIM measurements required dynamic range and exponents for the luminance, contrast, and structural term. The dynamic range was set to the HU range in the ground truth images, and the exponents were set to the typical value of 1. 

For each reconstructed image, we generated a single RMSE and a single SSIM value that was averaged within the VOI. For cases with multiple lesions, the metrics (RMSE and SSIM) were averaged across all lesions. With this approach, each participant had 200 scores for each metric (RMSE and SSIM). To combine the RMSE and SSIM scores, they were normalized to a range of 0 to 1 per disease type, with “0”being the worst measured value across all cases of the disease type for all received reconstructions. Similarly, “1” was assigned to the best measured value. For the RMSE results, the normalized values were subtracted from 1 such that the lowest RMSE (i.e., best result) was given a score of 1. The two normalized metrics were then averaged across the cases in each disease type to report a disease-specific score. The final ranking was determined by averaging the scores across the disease types. 

Although the True-CT challenge used only task-generic metrics (SSIM and RMSE), upon the completion of the challenge, further task-specific analysis was done on the lesion cases. The task-specific metrics were detectability index (d’) and accuracy of measuring lesion volume. 

The d’ was measured for each lesion in the reconstructed CT images based on a nonpre-whitening (NPW)-matched filter observer model, described in a previous study.[41] To measure d’, the lesion contrast and radius, patient-specific modulation transfer function (MTF), noise power spectrum (NPS), and global noise index (GNI) were extracted. The lesion contrast was measured by taking the difference of the average HU values between the foreground (i.e., lesion) and the background (parenchyma around the lesion). Lesion radius was determined based on the lesion volume and the sphericity assumption. The MTF, NPS, and GNI were calculated based on previously published techniques.[42–44] 

The other task-specific metric was the accuracy of measuring lesion volume. To determine the accuracy, lesion volume was measured from each reconstructed CT case and then compared against their corresponding ground truth values. Each lesion was segmented using a K-means segmentation algorithm. The segmented masks were then input to the Pyradiomics package[45] to calculate “mesh volume”. All segmented lesions were visually inspected. In case of failure in the segmentation, an alternative method, Geodesic Active 

Med Phys. Author manuscript; available in PMC 2025 April 07. 

Abadi et al. 

Page 8 

Contour,[46] was used to segment the lesions and to measure the lesion volume across all participants. The use of a consistent segmentation technique for all participants ensured a fair comparison between them. 

We further explored if the scoring metrics (SSIM, RMSE, d’, and lesion volume accuracy) provided complementary information about the image quality of the reconstructed images. This was done by correlating the scoring metrics using Pearson’s technique, assuming a normal distribution for the scoring metrics. Additionally, a statistical comparison was made between the top two performers for each scoring metric using a paired t-test, with the null hypothesis stating that the mean difference between the two algorithms is zero. The power of the test was calculated based on an effect size defined as the mean difference between the two algorithms divided by the standard deviation of the differences. A significance level of 5% was set for determining statistical significance. 

## **3 | RESULTS** 

Figure 2 shows example renditions of three human models, one for each disease type, illustrating both surface renderings and their corresponding digital ground truth. This figure highlights the realism, and the intricate detail of the structures included in the human models. 

The True-CT challenge attracted 52 participants, out of which five successfully completed the challenge and submitted the requested 200 reconstructions. Figure 3 showcases example reconstructions of three human models with different disease types from the five participants, referred in anonymized letters “A” to “E”. This figure demonstrates the variations in image quality among different submissions. Each participant reported that they fine-tuned their reconstruction parameters based on the disease type with a short description for their reconstruction technique described below. 

## **3.1 | Participant A: Model-based iterative reconstruction (MBIR)** 

Reconstructions were performed by implementing an MBIR technique.[47] This physics-based approach utilizes a noise model to characterize the noise inherent in data acquisition, and a prior image model to represent image noise and texture features. Additionally, a forward model was incorporated to account for key physical parameters, including focal spot positions, voxel and detector geometries, and the intersection lengths between rays and voxels. Informed by this physics model, a Consensus Equilibrium numerical method was employed to solve a joint optimization problem, estimating the maximum a posteriori solution by simultaneously fitting all models. This reconstruction technique did not incorporate any interpolation or rebinning to reduce image artifacts and improve spatial resolution. 

## **3.2 | Participant B: Katsevich algorithm** 

Participant “B” utilized the Katsevich reconstruction algorithm.[48] The algorithm was implemented following a previously published technique[49] except for computing the π -line. The π -line is a line segment that connects a point to be reconstructed in the helix and two points on the helix which are separated by less than one helical turn. The π -line was used 

Med Phys. Author manuscript; available in PMC 2025 April 07. 

Abadi et al. 

Page 9 

to determine which portion of data should be back-projected and calculated using the Izen technique,[50] accelerated with Graphical Processing Units (GPUs). 

## **3.3 | Participant C: Algebraic reconstruction** 

Reconstructions were done using the open-source Multi-Channel x-ray CT Reconstruction Toolkit.[39] The Toolkit implements matched, voxel-centric separable footprint CT reconstruction operators for projection and backprojection.[51] Approximate analytical reconstructions were first performed by cone-parallel rebinning followed by weighted filtered backprojection (WFBP) with a ramp filter.[52] To improve the data consistency of the reconstructions, unregularized, simultaneous algebraic reconstruction updates were then performed starting from the WFBP reconstructions and using the native sinogram data and geometry (prior to rebinning). Algebraic updates were performed with the BiCGSTAB(l) algorithm[53] using two iterations with three exclusive subsets of projections evaluated per iteration. 

## **3.4 | Participant D: Feldkamp-Davis-Kress reconstruction** 

This participant reconstructed the CT images using an Feldkamp-Davis-Kress (FDK) technique inspired by a previously established method.[54] The helical projections were rebinned into cone-parallel beams. The rebinned projections were preprocessed by applying pre-weighting and ramp-filtering. After preprocessing, a spiral cone-beam FDK reconstruction was performed, incorporating Parker weighting to address the angular incompleteness of the projections. This approach was applied to improve image quality by compensating for data redundancy and correcting potential artifacts. 

## **3.5 | Participant E: Fan-beam filtered backprojection** 

The reconstruction approach was based on a fan-beam filtered back-projection (FBP) algorithm.[55] To correct for cone beam artifacts, an additional weighting factor was applied while summing conjugate rays.[56] This factor effectively upweights conjugate rays with a smaller cone angle, reducing image artifacts due to off-axis rays. The strength of the cone beam correction and the cutoff frequency of the ramp filter were individually selected for each reconstruction task using a qualitative assessment. This resulted in cone beam weighting factors of 1.0,1.0, and 0.4 and ramp filter cutoffs at 90%,60%,and 40% of the Nyquist frequency for the emphysema, lung lesion, and liver lesion datasets, respectively. 

Figure 4 illustrates the normalized scores for each disease type and metric across all participants. The box plots show that the image quality was variable across patients and disease types. The patient-to-patient variability was higher in the emphysema cases compared to the lung and liver cases because the region of interest included the whole lungs, and the amount of emphysema was quite variable between the human models. The scores are summarized in Table 3. The overall scores demonstrate that participant “A” had the best performance in all categories, except for the metrics of d’ in the lung lesions and RMSE in the liver lesions. Participant “A” had an average score of 0.41 ± 0.22, 0.48 ± 0.32, and 0.42 ± 0.33 for the emphysema, lung lesion, and liver lesion cases, respectively. It should be noted that for the volume accuracy measurements, the K-means algorithm failed to segment 

Med Phys. Author manuscript; available in PMC 2025 April 07. 

Abadi et al. 

Page 10 

17 (out of 204) lung lesions and 27 (out of 246) liver lesions. As elaborated in the Methods, these lesions were segmented using the Geodesic Active Contour technique. 

Table 4 presents the correlation results among the measured image quality metrics across the disease types. In the emphysema cases, the correlation coefficient was 0.81 (p < 0.05) between the SSIM and RMSE measurements. Similarly, there was a high correlation between the SSIM and RMSE in the lung lesion cases with a correlation coefficient of 0.83. In contrast, the other metrics demonstrated low correlations coefficients, all below 0.17, showing that the d’ and volume accuracy metrics provided complementary quality evaluations. In the liver cases, the correlation coefficient between SSIM and d’ was 0.37 (p < 0.05), while the remaining metrics had correlation coefficients below 0.13. 

Table 5 summarizes the paired t-test comparisons between the top two performers for each metric and disease type. The results demonstrate that in all conditions, there was a statistically significant difference (p < 0.002) between the top two performers with a high power (>0.88). 

## **4 | DISCUSSION** 

CT reconstruction can be objectively assessed using virtual imaging tools, offering the unique advantage of access to a diverse population of diseased human models with known ground truth. Consequently, a virtual imaging framework was employed in this study to create a library of simulated CT projection images, providing a quantitative means for evaluating CT reconstruction methods. The utility of this framework was showcased in the AAPM True-CT reconstruction grand challenge where participants were provided with simulated projection images, which they reconstructed and returned for scoring. 

An interesting observation was that the quantitative scores varied for each participant within each disease type, underscoring the influence of patient attributes and disease severities on image quality. For instance, in emphysema cases, the RMSE and SSIM scores were strongly affected by disease severity (i.e., the emphysema percentage within the lungs). Emphysematous regions are less attenuative and more uniform compared to lung parenchyma, leading to lower attenuation and, consequently, lower RMSE values in more severe cases. A similar trend was observed for SSIM, with higher SSIM scores (indicating superior image quality) in severe cases. Similarly, d’ and volume accuracy were influenced by lesion contrast and size. Given these dependencies, a comprehensive and fair evaluation of CT reconstruction methods necessitates testing across a diverse range of patient attributes and disease severities. Our study employed a variety of task-generic and task-specific scores, well-established for image quality assessments. While some scores correlated with each other, others did not. These results offer two important realities: 1) Not all metrics are the same—differing metrics offer complementary roles and are needed for comprehensive image quality assessment; and 2) not all patients are the same—while aggregates across patients are necessary to provide average results for a given technology, any relevant clinical technology needs to be mindful of patient specificity in the practice of patient-centered care. 

Med Phys. Author manuscript; available in PMC 2025 April 07. 

Abadi et al. 

Page 11 

The scoring metrics used in this study were influenced by a combination of image quality attributes, each contributing differently to the performance evaluations. For instance, RMSE and SSIM are primarily sensitive to HU accuracy and image noise, while d’ is influenced by lesion contrast, spatial resolution, and noise. Morphological radiomics features, on the other hand, depend heavily on spatial resolution around the lesion. These attributes—HU accuracy, lesion contrast, image noise, and spatial resolution—are often interdependent, requiring trade-offs. For example, achieving high spatial resolution typically involves using sharp kernels, which may result in increased noise or compromised HU accuracy. Therefore, optimizing the balance between HU accuracy, noise magnitude, and spatial resolution is task dependent. This trade-off was evident in this challenge, where different reconstruction techniques showed varying performances depending on the metric evaluated. Participants who produced sharper reconstructions performed better in terms of lesion volume accuracy. However, for d’, achieving an optimal balance between noise and spatial resolution was more critical. 

In silico CT sinogram datasets, such as the one generated in this work, have numerous applications in development and evaluation of CT reconstruction techniques. With the full control over the attributes of the human models and acquisition parameters, researchers can systematically explore patient attributes or imaging conditions under which their technique may fail. These evaluations provide valuable insights into potential sources of error in CT reconstruction algorithms, enabling researchers to refine their methodologies. Additionally, such in silico CT sinogram datasets hold significant potential in advancing machine learning-based reconstruction techniques. As machine learning based algorithms continue to emerge, it is important to assess their applicability in CT image reconstructions. While datasets like the one introduced in this grand challenge could serve as valuable resources for such evaluations, we did not receive any submissions employing machine learning approaches during this challenge. This may be attributed to factors such as the limited timeframe of the challenge, the unavailability of training data, and the time-intensive nature of developing and fine-tuning machine learning models, especially when no prior training data are available. Additionally, the substantial size of the challenge dataset—comprising 200 CT sinograms of chest or abdomen scans—may have contributed to only 5 of the 52 registrants successfully completing the challenge. To support future evaluations of CT reconstruction methods, including those utilizing machine learning, interested researchers can obtain access to the acquired sinogram data and ground truth values by contacting the corresponding author of this article. 

A key lesson from organizing the True-CT challenge was the importance of standardized data formats for both CT projection and reconstruction data.[57] For the projection data, we adhered to the previously developed DICOM-CT-PD format, which enabled us to define private headers necessary for image reconstruction (e.g., NumberofDetectorRows/Columns, DetectorShape, ConstantRadialDistance, and NumberofSourceAngularSteps). Attaching this metadata directly to the projection images streamlines the reconstruction process and simplifies the data distribution process. While DICOM-CT-PD provides comprehensive acquisition and scanner geometry parameters in its header, processing thousands of DICOM images for one reconstruction can be computationally expensive using common tools (e.g., MATLAB) due to the overhead associated with opening each individual file. Additionally, 

Med Phys. Author manuscript; available in PMC 2025 April 07. 

Abadi et al. 

Page 12 

similar to the standard DICOM format, there is a large amount of duplicate data in headers from the same scan. To increase efficiency of both time and storage space, a threedimensional DICOM projection format may be considered in the future.[57] Similar needs can be envisioned for the reconstructed images, particularly for virtual data, so that image analytics can be given contingent access to certain information for targeted analysis. Such information may include the location and morphology of lesions, provided or de-identified per the goal of the analysis. Some of such added refinements are under deliberation by the ongoing efforts of the AAPM Task Group 387.[58] 

This study had several limitations. The CT reconstructions were evaluated using task-generic and task-specific scores, with each score being normalized across all participants per disease type, and the final score being an average of these scores weighted equally. However, this normalization and averaging approach may impact the overall ranking especially if the distribution of a particular score is skewed. In addition, averaging with equal weights may not necessarily reflect the images with the best image quality. Future studies may explore other strategies to combine image quality metrics tailored to targeted imaging applications and tasks. Further, our study focused on evaluating pathologies while future work may evaluate reconstruction techniques in terms of image quality for other anatomical regions such as airways, vasculature, and fine structures. Additionally, this study included mainly adult human models with no motions. Future studies may incorporate a broader range of age-based human models with cardiac, respiratory, or incidental motions, allowing for image reconstruction evaluations where motion is present. Another limitation is that this study modeled only one vendor-neutral CT scanner which represented current modern CT scanners. Future studies may expand on the diversity of the scanner models by including models of both legacy and state-of-the-art scanners. Lastly, this challenge demonstrated the utility of virtual imaging trials in evaluating CT reconstructions alone. Future studies may apply similar methodologies to evaluate other CT image formation processes, such as segmentation techniques, radiomics, and image quantification. 

## **5 | CONCLUSION** 

Virtual imaging trials enabled objective assessment of CT reconstructions with the unique advantage of access to a diverse population of diseased human models with known ground truth. This study highlights the significant potential of virtual imaging trials in objective assessment of imaging technologies and interventions. 

## **ACKNOWLEDGMENTS** 

This study was supported by the National Institutes of Health (P41EB028744, R01HL155293, and R01EB001838). The authors thank the AAPM’s working group on grand challenges for approving and facilitating this challenge. The authors thank Jeffrey Fessler, Joseph Lo, Raj Panta, Isabel Montero, Cindy McCabe, Anuj Kapadia, Greeshma Agasthya, Emily Townley, Benjamin Bearce, Sarah Gerard, and Samuel Armato for valuable discussions. The authors would like to express gratitude to COPDGene (U01 HL089897 and U01 HL089856 and by NIH contract 75N92023D00011) for generously providing the invaluable data used in this study. The COPDGene study (NCT00608764) has also been supported by the COPD Foundation through contributions made to an Industry Advisory Committee that has included AstraZeneca, Bayer Pharmaceuticals, Boehringer-Ingelheim, Genentech, GlaxoSmithKline, Novartis, Pfizer, and Sunovion. 

Med Phys. Author manuscript; available in PMC 2025 April 07. 

Abadi et al. 

Page 13 

## **REFERENCES** 

1. Szczykutowicz TP, Toia GV, Dhanantwari A, Nett B. A review of deep learning CT reconstruction: concepts, limitations, and promise in clinical practice. Current Radiol Rep. 2022;10(9):101–115. 

2. Wang G, Ye JC, Mueller K, Fessler JA. Image reconstruction is a new frontier of machine learning. IEEE Trans Med Imaging. 2018;37(6):1289–1296. [PubMed: 29870359] 

3. Ravishankar S, Ye JC, Fessler JA. Image reconstruction: from sparsity to data-adaptive methods and machine learning. Proc IEEE. 2019;108(1):86–109. 

4. McLeavy C, Chunara M, Gravell R, et al. The future of CT: deep learning reconstruction. Clin Radiol. 2021;76(6):407–415. [PubMed: 33637310] 

5. Stiller W Basics of iterative reconstruction methods in computed tomography: a vendor-independent overview. Eur J Radiol. 2018;109:147–154. [PubMed: 30527298] 

6. Beister M, Kolditz D, Kalender WA. Iterative reconstruction methods in X-ray CT. Physica Med. 2012;28(2):94–108. 

7. Abadi E, Segars WP, Tsui BM, et al. Virtual clinical trials in medical imaging: a review. J Med Imaging. 2020;7(4):042805. 

8. Truth-Based CT (TrueCT) Reconstruction Challenge. AAPM; 2023. Accessed March 13, 2023. https://www.aapm.org/GrandChallenge/TrueCT/default.asp 

9. Segars W, Bond J, Frush J, et al. Population of anatomically variable 4D XCAT adult phantoms for imaging research and optimization. Med Phys. 2013;40(4):043701. [PubMed: 23556927] 

10. Sauer TJ, Abadi E, Segars P, Samei E. Anatomically and physiologically informed computational model of hepatic contrast perfusion for virtual imaging trials. Med Phys. 2022;49(5):2938–2951. [PubMed: 35195901] 

11. Sauer TJ, Samei E. Modeling dynamic, nutrient-access-based lesion progression using stochastic processes.SPIE;2019:1193–1200. 

12. Abadi E, Jadick G, Lynch DA, Segars WP, Samei E. Emphysema quantifications with CT scan: assessing the effects of acquisition protocols and imaging parameters using virtual imaging trials. Chest. 2023;163(5):1084–1100. [PubMed: 36462532] 

13. Ghesu F-C, Georgescu B, Zheng Y, et al. Multi-scale deep reinforcement learning for real-time 3D-landmark detection in CT scans. IEEE Trans Pattern Anal Mach Intell. 2017;41(1):176–189. [PubMed: 29990011] 

14. Kratzke L, Mistry N, Möhler C, et al. DirectORGANS 2.0. 

15. Yang D,Xu D,Zhou SK,et al..Automatic Liver Segmentation Using an Adversarial Image-to-Image Network. Springer; 2017:507–515. 

16. Gerard SE, Patton TJ, Christensen GE, Bayouth JE, Reinhardt JM. FissureNet: a deep learning approach for pulmonary fissure detection in CT images. IEEE Trans Med Imaging. 2018;38(1):156–166. [PubMed: 30106711] 

17. Schroeder W, Martin KM, Lorensen WE. The Visualization ToolKit an Object-Oriented Approach to 3D Graphics. Prentice-Hall, Inc.; 1998. 

18. Sauer TJ, Bejan A, Segars P, Samei E. Development and CT image-domain validation of a computational lung lesion model for use in virtual imaging trials. Med Phys.2023;50(7):4366– 4378. [PubMed: 36637206] 

19. Cruickshank A, Stieler G, Ameer F. Evaluation of the solitary pulmonary nodule. Intern Med J. 2019;49(3):306–315. [PubMed: 30897667] 

20. Brown RS Jr. Asymptomatic liver mass. Gastroenterology. 2006;131(2):619–623. [PubMed: 16890613] 

21. Abadi E, Segars WP, Sturgeon GM, Harrawood B, Kapadia A, Samei E. Modeling “textured” bones in virtual human phantoms. IEEE Trans Radiat Plasma Med Sci. 2018;3(1):47–53. [PubMed: 31559375] 

22. Abadi E, Segars WP, Sturgeon GM, Roos JE, Ravin CE, Samei E. Modeling lung architecture in the XCAT series of phantoms: physiologically based airways, arteries and veins. IEEE Trans Med Imaging. 2017;37(3):693–702. 

Med Phys. Author manuscript; available in PMC 2025 April 07. 

Abadi et al. 

Page 14 

23. Abadi E, Sturgeon GM, Agasthya G, et al. Airways, vasculature, and interstitial tissue: anatomically informed computational modeling of human lungs for virtual clinical trials. Med Imaging 2017 Phys Med Imaging. 2017;10132:101321Q. doi: 10.1117/12.2254739 

24. White DR, Griffith RV, Wilson IJ. “ICRU reports.” Reports of the International Commission on Radiation Units and Measurements. 1992(1):203–205. 

25. Abadi E, Harrawood B, Sharma S, Kapadia A, Segars WP, Samei E. DukeSim: a realistic, rapid, and scanner-specific simulation framework in computed tomography. IEEE Trans Med Imaging. 2018;38(6):1457–1465. [PubMed: 30561344] 

26. Abadi E, Harrawood B, Rajagopal JR, et al. Development of a scanner-specific simulation framework for photon-counting computed tomography. Biomed Phys Eng Express.2019;5(5):055008. [PubMed: 33304618] 

27. Sharma S, Abadi E, Kapadia A, Segars WP, Samei E. A GPU-accelerated framework for rapid estimation of scanner-specific scatter in CT for virtual imaging trials. Phys Med Biol. 2021;66(7):075004. 

28. Sharma S, Kapadia A, Fu W, Abadi E, Segars WP, Samei E. A real-time Monte Carlo tool for individualized dose estimations in clinical CT. Phys Med Biol. 2019;64(21):215020. [PubMed: 31539892] 

29. Jadick G, Abadi E, Harrawood B, Sharma S, Segars WP, Samei E. A scanner-specific framework for simulating CT images with tube current modulation. Physics in Medicine & Biology. 2021;66(18):185010. 

30. Shankar SS, Jadick GL, Hoffman EA, et al. Scanner-specific validation of a CT simulator using a COPD-emulated anthropomorphic phantom. SPIE; 2022:953–960. 

31. Abadi E, McCabe C, Harrawood B, Sotoudeh-Paima S, Segars WP, Samei E. Development and clinical applications of a virtual imaging framework for optimizing photon-counting CT. SPIE; 2022:426–432. 

32. Lecoq P Development of new scintillators for medical applications. Nucl Instrum Methods Phys Res A. 2016;809:130–139. 

33. Van Eijk CW. Inorganic scintillators in medical imaging. Phys Med Biol. 2002;47(8):R85. [PubMed: 12030568] 

34. FitzGerald P, Araujo S, Wu M, De Man B. Semiempirical, parameterized spectrum estimation for x-ray computed tomography. Med Phys. 2021;48(5):2199–2213. [PubMed: 33426704] 

35. Boone JM. Method for evaluating bow tie filter angle-dependent attenuation in CT: theory and simulation results. Med Phys. 2010;37(1):40–48. [PubMed: 20175464] 

36. Chen B, Leng S, Yu L, Holmes IIID, Fletcher J, McCollough C. An open library of CT patient projection data. SPIE; 2016:330–335. 

37. McCollough CH, Bartley AC, Carter RE, et al. Low-dose CT for the detection and classification of metastatic liver lesions: results of the 2016 low dose CT grand challenge. Med Phys. 2017;44(10):e339–e352. [PubMed: 29027235] 

38. Abràmoff MD, Magalhães PJ, Ram SJ. Image processing with ImageJ. Biophotonics Int. 2004;11(7):36–42. 

39. Clark DP, Badea CT. MCR toolkit: a GPU-based toolkit for multichannel reconstruction of preclinical and clinical x-ray CT data. Med Phys. 2023;50(8):4775–4796. [PubMed: 37285215] 

40. The Medical Challenge Initiative. 2024. https://github.com/QTIM-Lab/MedICI 

41. Smith TB, Solomon J, Samei E. Estimating detectability index in vivo: development and validation of an automated methodology. J Med Imaging. 2018;5(3):031403–031403. 

42. Smith TB, Abadi E, Sauer TJ, Fu W, Solomon J, Samei E. Development and validation of an automated methodology to assess perceptual in vivo noise texture in liver CT. J Med Imaging. 2021;8(5):052113. 

43. Christianson O, Winslow J, Frush DP, Samei E. Automated technique to measure noise in clinical CT examinations. Am J Roentgenol. 2015;205(1):W93–W99. [PubMed: 26102424] 

44. Sanders J, Hurwitz L, Samei E. Patient-specific quantification of image quality: an automated method for measuring spatial resolution in clinical CT images. Med Phys. 2016;43(10):5330– 5338. [PubMed: 27782718] 

Med Phys. Author manuscript; available in PMC 2025 April 07. 

Page 15 

Abadi et al. 

45. Van Griethuysen JJ, Fedorov A, Parmar C, et al. Computational radiomics system to decode the radiographic phenotype. Cancer Res. 2017;77(21):e104–e107. [PubMed: 29092951] 

46. Marquez-Neila P, Baumela L, Alvarez L. A morphological approach to curvature-based evolution of curves and surfaces. IEEE Trans Pattern Anal Mach Intell. 2013;36(1):2–17. 

47. Wang X, MacDougall RD, Chen P, Bouman CA, Warfield SK. Physics-based iterative reconstruction for dual-source and flying focal spot computed tomography. Med Phys. 2021;48(7):3595–3613. [PubMed: 33982297] 

48. Katsevich A An improved exact filtered backprojection algorithm for spiral computed tomography. Adv Appl Math. 2004;32(4):681–697. 

49. Noo F, Pack J, Heuscher D. Exact helical reconstruction using native cone-beam geometries. Phys Med Biol. 2003;48(23): 3787. [PubMed: 14703159] 

50. Izen S A fast algorithm to compute the π -line through points inside a helix cylinder. Proc Amer Math Soc. 2007;135(1):269–276. 

51. Long Y, Fessler JA, Balter JM. 3D forward and back-projection for X-ray CT using separable footprints. IEEE Trans Med Imaging. 2010;29(11):1839–1850. [PubMed: 20529732] 

52. Stierstorfer K, Rauscher A, Boese J, Bruder H, Schaller S, Flohr T. Weighted FBP—a simple approximate 3D FBP algorithm for multislice spiral CT with good dose usage for arbitrary pitch.Phys Med Biol. 2004;49(11):2209. [PubMed: 15248573] 

53. Sleijpen GL, Van Gijzen MB. Exploiting BiCGstab ( ℓ ) strategies to induce dimension reduction. SIAM J Sci Comput. 2010;32(5):2687–2709. 

54. Kong H, Liu R, Pan J, Yu H. Evaluation of an analytic reconstruction method as a platform for spectral cone-beam CT. IEEE Access. 2018;6:21314–21323. [PubMed: 30510887] 

55. Hsieh J Computed tomography: principles, design, artifacts, and recent advances. 2003. 

56. Tang X, Hsieh J, Hagiwara A, Nilsen RA, Thibault J-B, Drapkin E. A three-dimensional weighted cone beam filtered backprojection (CB-FBP) algorithm for image reconstruction in volumetric CT under a circular source trajectory. Phys Med Biol. 2005;50(16):3889. [PubMed: 16077234] 

57. Abadi E, Barufaldi B, Lago M, et al. Toward widespread use of virtual trials in medical imaging innovation and regulatory science. Med Phys. 2024. 

58. AAPM Task Group No. 387 - Consensus Recommendations on reliable development and use of Virtual Imaging Trials. AAPM; 2024. https://www.aapm.org/org/structure/default.asp? committee_code=TG387 

Med Phys. Author manuscript; available in PMC 2025 April 07. 

Page 16 

Abadi et al. 

## **FIGURE 1.** 

Framework of the True-CT challenge. 

Med Phys. Author manuscript; available in PMC 2025 April 07. 

Abadi et al. 

Page 17 

## **FIGURE 2.** 

Surface and voxelized renditions of three developed human models with emphysema (top row), lung lesion (middle row), and liver lesion (bottom row). The lung and liver lesions are highlighted with a green box. The voxelized images represent an axial cross-section of their corresponding models with intensities set to be their theoretical CT number (HU values) at the effective energy of 64 keV. The window level and width are −600 and 750 HU for the chest models, and 60 and 160 HU for the liver model. HU, Hounsfield Units. 

Med Phys. Author manuscript; available in PMC 2025 April 07. 

Abadi et al. 

Page 18 

**FIGURE 3.** 

Example reconstructions of human models with three disease types, prepared by the participants “A” to “E” along with the corresponding ground truth model. The window level and width are −600 HU and 750 HU for the emphysema and lung lesion cases. The liver lesion cases are shown with a window level and width of 60 HU and 160 HU. HU, Hounsfield Units. 

Med Phys. Author manuscript; available in PMC 2025 April 07. 

Abadi et al. 

Page 19 

## **FIGURE 4.** 

Boxplots showing normalized score per participant “A-E” and disease type. The scores were (a) SSIM, (b) RMSE, (c) detectability index (d’), and (d) volume accuracy. The scores are normalized to 0 and 1 per disease type so that the scores are comparable. The RMSE and volume accuracy are reported as “1-score” so that the higher values show better image quality. RMSE, root mean squared error; SSIM, structural similarity matrix. 

Med Phys. Author manuscript; available in PMC 2025 April 07. 

Abadi et al. 

Page 20 

**TABLE 1** 

Geometrical attributes of Duke-1 CT system. 

|Number of detector columns and rows|900 × 64|
|---|---|
|Number of projections per rotation|1000|
|Detector element size [mm]|1 × 1|
|Focal spot size [mm]|1 × 1|
|Source-to-isocenter distance [mm]|575|
|Source-to-detector distance [mm]|1075|



Med Phys. Author manuscript; available in PMC 2025 April 07. 

Abadi et al. 

Page 21 

**TABLE 2** 

Requested settings and format for the True-CT reconstruction CT data. 

|Field of view [mm]|500|
|---|---|
|In-plane pixel size [mm]|500/512|
|Slice thickness[mm]|chest cases: 0.55 liver cases: 1.5|
|Slice interval|= slice thickness|
|Data format|DICOM|
|No overlaps in the reconstructed data||
|Full reconstruction of the “z” field of view||



Med Phys. Author manuscript; available in PMC 2025 April 07. 

Abadi et al. 

Page 22 

|**Norm. score**<br>**A**<br>**B**<br>**C**<br>**D**<br>**E**<br>**abs. [min—max]**|**Emphysema**<br>SSIM<br>**0.29 ± 0.22**<br>0.26 ± 0.23<br>0.22 ± 0.23<br>0.19 ± 0.23<br>0.17 ± 0.22<br>0.22–0.78 (unitless)<br>1-RMSE<br>**0.54 ± 0.13**<br>0.50 ± 0.12<br>0.40 ± 0.13<br>0.26 ± 0.11<br>0.37 ± 0.12<br>77.6–186.7 (HU)<br>d’<br>N/A<br>1-Vol. Acc.<br>N/A<br>Avg.<br>**0.41 ± 0.22**<br>0.38 ± 0.22<br>0.31 ± 0.20<br>0.22 ± 0.18<br>0.27 ± 0.20<br>N/A<br>**Lung lesions**<br>SSIM<br>**0.71 ± 0.12**<br>0.53 ± 0.12<br>0.37 ± 0.11<br>0.19 ± 0.11<br>0.18 ± 0.11<br>0.14–0.42<br>1-RMSE<br>**0.85 ± 0.09**<br>0.76 ± 0.09<br>0.72 ± 0.08<br>0.58 ± 0.10<br>0.52 ± 0.14<br>251.6–490.5<br>d’<br>0.22 ± 0.13<br>0.25 ± 0.17<br>**0.33 ± 0.21**<br>0.26 ± 0.19<br>0.23 ± 0.17<br>6.7 – 64.6 (unitless)<br>1-Vol. Acc.<br>**0.94 ± 0.06**<br>0.86 ± 0.11<br>0.81 ± 0.17<br>0.77 ± 0.21<br>0.74 ± 0.22<br>1.75–616.6 (mm3)<br>Avg.<br>**0.48 ± 0.32**<br>0.43 ± 0.29<br>0.40 ± 0.27<br>0.36 ± 0.26<br>0.34 ± 0.20<br>N/A<br>**Liver lesions**<br>SSIM<br>**0.84 ± 0.12**<br>0.60 ± 0.18<br>0.70 ± 0.15<br>0.64 ± 0.17<br>0.75 ± 0.13<br>0.53–0.90 (unitless)<br>1-RMSE<br>0.75 ± 0.14<br>0.73 ± 0.14<br>**0.85 ± 0.13**<br>0.84 ± 0.13<br>0.59 ± 0.23<br>286.7–490.5 (HU)<br>d’<br>**0.38 ± 0.18**<br>0.10 ± 0.06<br>0.18 ± 0.10<br>0.10 ± 0.06<br>0.21 ± 0.10<br>0.1–8.2 (unitless)<br>1-Vol. Acc.<br>**0.82 ± 0.15**<br>0.71 ± 0.21<br>0.74 ± 0.19<br>0.70 ± 0.22<br>0.75 ± 0.18<br>1.2–753.1 (mm3)<br>Avg.<br>**0.42 ± 0.33**<br>0.38 ± 0.29<br>0.41 ± 0.32<br>0.40 ± 0.31<br>0.38 ± 0.30<br>N/A|
|---|---|



Med Phys. Author manuscript; available in PMC 2025 April 07. 

Abadi et al. 

Page 23 

**TABLE 4** 

Pearson correlation coefficients between the measured image quality metrics for each disease type. 

||**Emphysema**<br>**SSIM**<br>**RMSE**|**Lung lesions**<br>**SSIM**<br>**RMSE**<br>**d'**<br>**Vol. Acc.**|**Liver lesions**|
|---|---|---|---|
||||**SSIM**<br>**RMSE**<br>**d'**<br>**Vol. Acc.**|
|SSIM<br>RMSE<br>d'<br>Vol. Acc.|1<br>0.81*<br>1|1<br>0.88*<br>−0.06<br>−0.17*<br>1<br>−0.02<br>−0.10<br>1<br>0.00<br>1|1<br>0.04<br>0.37*<br>0.04<br>1<br>−0.02<br>−0.13*<br>1<br>−0.13*<br>1|



Note: The * denotes correlations with p-values below 0.05. 

Abbreviations: d’,detectability index; RMSE: root mean square; SSIM: structural similarity index; Vol.Acc.: lesion volume accuracy. 

Med Phys. Author manuscript; available in PMC 2025 April 07. 

Abadi et al. 

Page 24 

## **TABLE 5** 

Paired t-test results comparing the top two participants for each scoring metric and disease type. 

|||**Top two participants**|**_p_-value**|**Power**|
|---|---|---|---|---|
|**Emphysema**|SSIM|A and B|<0.001|1.00|
||1-RMSE|A and B|<0.001|1.00|
||d'|N/A|||
||1-Vol. Acc.|N/A|||
|**Lung lesions**|SSIM|A and B|<0.001|1.00|
||1-RMSE|A and B|<0.001|1.00|
||d'|C and B|<0.001|1.00|
||1-Vol. Acc.|A and B|<0.001|1.00|
|**Liver lesions**|SSIM|A and E|<0.001|1.00|
||1-RMSE|C and D|0.002|0.88|
||d'|A and E|<0.001|1.00|
||1-Vol. Acc.|A and E|<0.001|0.98|



Abbreviations: d’, detectability index; SSIM, structural similarity index; RMSE, root mean square; Vol. Acc., lesion volume accuracy. 

Med Phys. Author manuscript; available in PMC 2025 April 07. 

