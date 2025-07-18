# 🔉 Sound Event Localization and Detection

Final project for the course Deep Learning based on the DCASE 2023 Challenge -
**Sound Event Localization and Detection Evaluated in Real Spatial Sound
Scenes**.

## Description

Given multichannel audio input, a sound event localization and detection
(SELD) system outputs a temporal activation track for each of the target sound
classes, along with one or more corresponding spatial trajectories when the
track indicates activity. This results in a spatio-temporal characterization
of the acoustic scene that can be used in a wide range of machine cognition
tasks, such as inference on the type of environment, self-localization,
navigation with occluded targets, tracking of specific types of sound sources,
smart-home applications, scene visualization systems, and acoustic monitoring,
among others.

*The text in this section is excerpted from the DCASE2023 Challenge.*

## Dataset

The Sony-TAu Realistic Spatial Soundscapes 2023 (STARSS23) dataset contains
multichannel recordings of sound scenes in various rooms and environments,
together with temporal and spatial annotations of prominent events belonging
to a set of target classes. The dataset is collected in two different sites,
in Tampere, Finland by the Audio Researh Group (ARG) of Tampere University,
and in Tokyo, Japan by Sony, using a similar setup and annotation procedure.
As in the previous challenges, the dataset is delivered in two spatial
recording formats.

*The preceding text in this section is excerpted from the DCASE2023 Challenge.*

Dataset should be placed in the `data` directory in the root of the project.
The directory structure should look like:

```text
data
├── foa_dev
│   ├── dev-test-sony
│   ├── dev-test-tau
│   ├── dev-train-sony
│   └── dev-train-tau
├── metadata_dev
│   ├── dev-test-sony
│   ├── dev-test-tau
│   ├── dev-train-sony
│   └── dev-train-tau
└── video_dev
    ├── dev-test-sony
    ├── dev-test-tau
    ├── dev-train-sony
    └── dev-train-tau
```

## Results

The results of the evaluation are shown in the report in this repository. An
example of labels and predictions for a windows of 5 seconds is shown below.

<img src="figures/predictions.png" width="500" alt="Example of predictions">

## References

- [DCASE2023 Challenge](https://dcase.community/challenge2023/)
- [STARSS23: Sony-TAu Realistic Spatial Soundscapes 2023](https://zenodo.org/records/7880637)
