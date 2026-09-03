\# SEN2SR-Lite



\## Deep Learning Based Super Resolution Mapping from Medium Resolution Satellite Imagery



SEN2SR-Lite is a deep-learning-based satellite image super-resolution system designed to enhance medium-resolution Sentinel-2 imagery while preserving useful spectral information for downstream agricultural analysis.



The system processes Sentinel-2 bands B02, B03, B04 and B08 and performs 4× spatial super-resolution from 10 m input resolution to an approximately 2.5 m output scale using deep learning models.



\---



\## SIH 2026 Problem



Medium-resolution satellite imagery is widely available for large-area monitoring, but its spatial resolution can make it difficult to identify small buildings, narrow roads, field boundaries and localized land-surface changes.



This project implements a deep-learning-based Super Resolution Mapping (SRM) workflow to enhance spatial detail while maintaining spectral consistency for analysis.



\---



\## Project Highlights



\- 4-band Sentinel-2 super-resolution

\- Sentinel-2 B02, B03, B04 and B08 bands

\- 10 m → 2.5 m equivalent 4× enhancement

\- EDSR, SwinIR and ESRGAN model comparison

\- Bicubic interpolation baseline

\- Uncertainty-aware reconstruction analysis

\- Spectral consistency evaluation

\- High-resolution reference validation

\- NDVI analysis

\- Vegetation health assessment

\- Dry/barren area detection

\- Agricultural area estimation

\- Agricultural feature extraction

\- RGB image super-resolution branch

\- Integrated processing pipeline



\---



\## System Pipeline



```text

Sentinel-2 Medium Resolution Image

&#x20;             |

&#x20;             v

&#x20;    B02 / B03 / B04 / B08

&#x20;             |

&#x20;             v

&#x20;       Preprocessing

&#x20;             |

&#x20;             v

&#x20;  +-------------------------+

&#x20;  | Super Resolution Models |

&#x20;  |-------------------------|

&#x20;  | Bicubic                 |

&#x20;  | EDSR                    |

&#x20;  | SwinIR                  |

&#x20;  | ESRGAN                  |

&#x20;  +-------------------------+

&#x20;             |

&#x20;             v

&#x20;       Enhanced SR Image

&#x20;             |

&#x20;             v

&#x20;      Spectral Analysis

&#x20;             |

&#x20;             v

&#x20;           NDVI

&#x20;             |

&#x20;             v

&#x20;  Agricultural Intelligence

&#x20;             |

&#x20;     +-------+--------+

&#x20;     |       |        |

&#x20;     v       v        v

&#x20;Vegetation Dry/Barren Agricultural

&#x20; Health     Detection     Area

&#x20;             |

&#x20;             v

&#x20;   Feature Extraction

&#x20;             |

&#x20;             v

&#x20;Crop Growth / Condition

&#x20;     / Statistics



Models Used

1\. Bicubic Interpolation



Bicubic interpolation is used as the baseline method for comparison with the deep-learning-based super-resolution models.



2\. EDSR



Enhanced Deep Residual Network (EDSR) is implemented as the primary convolutional super-resolution model.



Configuration used in this project:



Input channels: 4

Output channels: 4

Scale factor: 4×

Feature channels: 64

Residual blocks: 16

Residual scaling: 0.1

Upsampling: PixelShuffle

Training loss: L1

Optimizer: Adam

3\. SwinIR



A simplified SwinIR-style architecture is implemented for transformer-based super-resolution comparison.



Configuration includes:



Input channels: 4

Output channels: 4

Scale factor: 4×

Feature embedding: 64

Transformer depth: 4

Attention heads: 4

Window size: 8

4\. ESRGAN



An RRDB-style ESRGAN generator is implemented for deep-learning-based comparison.



The current experimental implementation uses L1 reconstruction training.



5\. RGB EDSR



A separate RGB EDSR branch is implemented for super-resolution of standard RGB satellite images.



The RGB training data is derived from the Sentinel-2 spectral dataset by using:



B04 → Red

B03 → Green

B02 → Blue



The RGB branch performs:



RGB Image → RGB EDSR → 4× RGB Super-Resolution



For arbitrary JPG/PNG images, the result represents 4× visual super-resolution. The approximately 2.5 m geospatial interpretation applies specifically to the Sentinel-2 10 m input branch.



Dataset



The project uses the SEN2NAIPv2 dataset.



Dataset structure used locally:



dataset/

└── local/

&#x20;   ├── train/

&#x20;   │   ├── lr/

&#x20;   │   └── hr/

&#x20;   ├── val/

&#x20;   │   ├── lr/

&#x20;   │   └── hr/

&#x20;   └── test/

&#x20;       ├── lr/

&#x20;       └── hr/



The local experimental dataset contains:



100 training pairs

20 validation pairs

20 test pairs

Input Data

4 × 130 × 130

High-Resolution Reference

4 × 520 × 520

Spectral Bands

B02 - Blue

B03 - Green

B04 - Red

B08 - Near Infrared

Experimental Results



Evaluation was performed on 20 local test samples.



Model	PSNR (dB)	SSIM

Bicubic Baseline	27.3894	0.7028

EDSR	24.1201	0.5382

SwinIR	22.5216	0.4393

ESRGAN	19.9297	0.4320



Among the implemented deep-learning models, EDSR achieved the best PSNR and SSIM in the current local evaluation.



Bicubic is reported separately as the interpolation baseline and is not a deep-learning model.



Spectral Consistency Analysis



The system evaluates spectral consistency of the super-resolved imagery using the four spectral bands:



B02 - Blue

B03 - Green

B04 - Red

B08 - NIR



The analysis includes:



Normalized spectral error

Spectral correlation

Band-wise comparison



This helps evaluate whether the super-resolution process maintains useful spectral characteristics.



Uncertainty Analysis



The project includes an uncertainty-analysis module based on multiple noisy EDSR predictions.



Multiple predictions are generated for the same input and their variation is used as a practical uncertainty proxy.



The module generates:



Mean super-resolution output

Pixel-wise uncertainty

Average uncertainty map

Confidence representation



This provides an uncertainty estimate for reconstruction analysis.



High-Resolution Validation



The super-resolved output is compared against the high-resolution reference imagery available in the SEN2NAIPv2 dataset.



The validation includes:



PSNR

SSIM

Band-wise comparison

Average reconstruction performance

NDVI Analysis



The enhanced Red and Near-Infrared bands are used to calculate NDVI.



The formula used is:



NDVI = (NIR - Red) / (NIR + Red)



where:



NIR = B08

Red = B04



The generated NDVI maps are used for further agricultural analysis.



Vegetation Health Analysis



Vegetation health is classified using NDVI ranges.



The implemented classification includes:



NDVI < 0       → Water

0.0 – 0.2      → Bare / Non-vegetated

0.2 – 0.4      → Sparse / Stressed vegetation

0.4 – 0.6      → Moderate vegetation

NDVI >= 0.6    → Healthy vegetation

Dry and Barren Area Detection



The system identifies low-vegetation and dry/barren areas using NDVI-based thresholds.



The analysis distinguishes:



Water

Dry/barren areas

Very low vegetation

Sparse vegetation

Vegetated areas

Agricultural Area Detection



Agricultural area estimation is performed using NDVI-based vegetation thresholds.



The system calculates:



Agricultural area percentage

Healthy vegetation percentage

Sparse vegetation percentage

Dry/low vegetation percentage

Mean NDVI

Agricultural Feature Extraction



The agricultural feature extraction module generates statistics and indicators from the NDVI results.



Extracted features include:



Mean NDVI

Median NDVI

Minimum NDVI

Maximum NDVI

NDVI standard deviation

Agricultural area percentage

Healthy vegetation percentage

Moderate vegetation percentage

Low vegetation percentage

Dry/barren percentage

Water percentage

Vegetation level

Agricultural condition

Crop growth indication

Integrated Processing Pipeline



The complete workflow is integrated through:



src/main\_pipeline.py



The pipeline combines:



Model Evaluation

&#x20;       ↓

Super-Resolution Visualization

&#x20;       ↓

High-Resolution Validation

&#x20;       ↓

Spectral Consistency

&#x20;       ↓

Uncertainty Analysis

&#x20;       ↓

NDVI Analysis

&#x20;       ↓

Vegetation Health

&#x20;       ↓

Dry/Barren Detection

&#x20;       ↓

Agricultural Area Detection

&#x20;       ↓

Agricultural Feature Extraction

Project Structure

SEN2SR-Lite/

│

├── dashboard/

│   └── app.py

│

├── dataset/

│   ├── local/

│   └── current/

│

├── models/

│   ├── edsr\_local.pth

│   ├── swinir\_local.pth

│   ├── esrgan\_local.pth

│   └── rgb\_edsr\_local.pth

│

├── results/

│

├── src/

│   ├── edsr\_model.py

│   ├── swinir\_model.py

│   ├── esrgan\_model.py

│   ├── rgb\_edsr\_model.py

│   │

│   ├── trains.py

│   ├── train\_swinir.py

│   ├── train\_esrgan.py

│   ├── train\_rgb\_edsr.py

│   │

│   ├── evaluate\_all\_models.py

│   ├── evaluate\_sr.py

│   ├── evaluate\_swinir.py

│   ├── high\_resolution\_validation.py

│   ├── spectral\_consistency.py

│   ├── uncertainty\_map.py

│   │

│   ├── ndvi\_analysis.py

│   ├── vegetation\_health.py

│   ├── dry\_barren\_detection.py

│   ├── agricultural\_area\_detection.py

│   ├── agricultural\_feature\_extraction.py

│   │

│   ├── dataset\_loader.py

│   ├── data\_preprocess.py

│   ├── download\_samples.py

│   ├── generate\_edsr.py

│   ├── infer\_current\_sentinel.py

│   ├── visualize\_sr\_results.py

│   └── main\_pipeline.py

│

├── requirements.txt

├── .gitignore

└── README.md

Technology Stack

Programming

Python

Deep Learning

PyTorch

Torchvision

Image and Geospatial Processing

Rasterio

NumPy

Pillow

scikit-image

Data Processing

Pandas

Visualization

Matplotlib

Development Tools

Visual Studio Code

Python Virtual Environment

Git

GitHub

Installation



Clone the repository:



git clone https://github.com/kusuma-s46/SEN2SR-Lite.git

cd SEN2SR-Lite



Create a virtual environment:



Windows

python -m venv venv



Activate the environment:



venv\\Scripts\\activate



Install the required packages:



pip install -r requirements.txt

Running the Project



Run the integrated pipeline from the project root:



python src\\main\_pipeline.py



Individual modules can also be executed from the project root.



For example:



python src\\evaluate\_all\_models.py

Model Weights



The trained model weights are intentionally excluded from GitHub using .gitignore.



The trained models used locally are:



edsr\_local.pth

swinir\_local.pth

esrgan\_local.pth

rgb\_edsr\_local.pth



The model files remain available in the local project environment but are not committed to the GitHub repository.



Research Focus



The project combines satellite image super-resolution with spectral analysis and agricultural intelligence.



The core research focus is:



Spatial Enhancement

&#x20;       +

Spectral Consistency

&#x20;       +

Uncertainty Analysis

&#x20;       +

Agricultural Intelligence



This provides a complete workflow from medium-resolution satellite imagery enhancement to downstream agricultural assessment.



Reference Resources



· https://huggingface.co/datasets/tacofoundation/SEN2NAIPv2



· https://www.nature.com/articles/s41597-024-04214-y



· https://github.com/sanghyun-son/EDSR-PyTorch



· https://github.com/JingyunLiang/SwinIR



· https://github.com/xinntao/ESRGAN



· https://dataspace.copernicus.eu/



Project Information



Project: SEN2SR-Lite



Problem: Deep Learning Based Super Resolution Mapping from Medium Resolution Satellite Imagery



Event: Smart India Hackathon 2026



Framework: PyTorch



Primary Model: EDSR



Input: Sentinel-2 B02, B03, B04, B08



Super-Resolution Scale: 4×



Input Resolution: 10 m



Output Scale: Approximately 2.5 m



Dataset: SEN2NAIPv2







