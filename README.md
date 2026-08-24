## Project Author and Mechanical Design

This project was independently designed and developed by **Dongju Lee**. The work includes the project concept, system design, hardware integration, mechanical assembly, custom 3D modelling, Arduino firmware, Python runtime, AI pipeline, App Lab dashboard, testing, and documentation.

All custom mechanical parts used in the project—including the robot chassis, base plate, ultrasonic sensor mounts, camera mount, and electronic component mounts—were modelled by the project author, except for the TT motor brackets.

The TT motor brackets use the external **[TT Gear Motor Mount](https://makerworld.com/en/models/519327-tt-gear-motor-mount)** model created by **[Chief human](https://makerworld.com/en/@Chief_human)**. The model is published under the **[Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/)**.

The creator also provided direct written permission to use and modify the model for this project and requested attribution through the MakerWorld model link. This project does not claim ownership of the original TT motor bracket design. No modification to the original bracket geometry is documented in this repository.

## Credits and Third-Party Assets

| Asset / Dependency | Repository Evidence | License / Credit Status |
|---|---|---|
| MobileNet SSD ONNX person detector | `mpu/ai/models/ssd_mobilenet_v1_12.onnx` | <!-- TODO: Add verified source URL and license. --> |
| SHEL5K safety helmet dataset | `mpu/config.py`, `mpu/ai/dataset/loader.py` | <!-- TODO: Add verified dataset source URL, citation, and license/terms. --> |
| PyTorch / torchvision pretrained weights | `mpu/ai/train.py` | <!-- TODO: Add competition-required third-party license attribution if needed. --> |
| Python libraries | `requirements.txt`, `app_lab/requirements_app_lab.txt` | See each package license. |
| Custom CAD / 3D model files | `STL/*.stl`, `STL/*.3mf` excluding the TT motor bracket | Created by Dongju Lee |
| TT motor bracket model | [TT Gear Motor Mount](https://makerworld.com/en/models/519327-tt-gear-motor-mount) by [Chief human](https://makerworld.com/en/@Chief_human) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); creator also provided direct permission to use and modify the model |
| Robot photographs, dashboard screenshots, circuit image, video | `docs/images/` | Created by Dongju Lee for this project |

The TT motor bracket attribution and license notice must be preserved when the model or a modified version is redistributed. Any future modifications must be clearly identified as changes to the original model.

## License

Except for the third-party assets explicitly identified below, the source code, documentation, photographs, videos, circuit diagram, and original 3D models created for this project are **Copyright © 2026 Dongju Lee. All rights reserved.**

No open-source or Creative Commons license is granted for the original work created by Dongju Lee unless a separate written license is added later.

### Separately Licensed Third-Party Asset

The **[TT Gear Motor Mount](https://makerworld.com/en/models/519327-tt-gear-motor-mount)** was created by **[Chief human](https://makerworld.com/en/@Chief_human)** and is licensed separately under the **[Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/)**.

The CC BY 4.0 license applies only to the TT Gear Motor Mount. It does not apply to the source code, documentation, photographs, videos, circuit diagram, or original 3D models created by Dongju Lee.

Other third-party datasets, AI models, libraries, and dependencies remain subject to their respective licenses and terms.