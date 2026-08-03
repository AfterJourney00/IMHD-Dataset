

# IMHD$`^2`$: Conjunto de datos de interacciones humano-objeto altamente dinámicas con inercia y múltiples vistas

[![arXiv](https://img.shields.io/badge/arXiv-2312.08869-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2312.08869) [![project page](https://img.shields.io/badge/project_page-blue)](https://afterjourney00.github.io/IM-HOI.github.io/)
<!-- ![visitor badge](https://visitor-badge.laobi.icu/badge?page_id=AfterJourney00.IMHD-Dataset&left_color=red&right_color=green&left_text=visitors) -->


> **I'M HOI: Captura monocular consciente de la inercia de interacciones 3D humano-objeto**  
> *Chengfeng Zhao, Juze Zhang, Jiashen Du, Ziwei Shan, Junye Wang, Jingyi Yu, Jingya Wang, Lan Xu\**

****

<p align="center">
<img src="assets/dataset_gallery_full.png" alt="teaser" width="8128"/>
</p> 

## 🔥Noticias
- ***Enero, 2026:*** 🔈🔈 IMHD$`^2`$ ahora está completamente archivado en HuggingFace, consulta y descárgalo [aquí](https://huggingface.co/datasets/AfterJourney/IMHD-Dataset).
- ***Febrero, 2025:*** 🔈🔈 IMHD$`^2`$ se une al [3er Reto Rhobin](https://rhobin-challenge.github.io/), ¡bienvenidos a participar!
- ***Noviembre, 2024:*** 🔈🔈 ¡Descarga o consulta en línea nuestros [videos](https://pan.quark.cn/s/e205d3c0e072) y [segmentaciones](https://pan.quark.cn/s/a1662d344e4a)!
- ***Septiembre, 2024:*** 🔈🔈 ¡Se han publicado las segmentaciones a nivel de instancia!
- ***Julio, 2024:*** 🔈🔈 ¡Se han publicado los videos sin procesar!
- ***Mayo, 2024:*** 🔈🔈 ¡Se han publicado los puntos clave humanos 2D y 3D de 32 vistas!
- ***Marzo, 2024:*** 🎉🎉 [I'M HOI](https://afterjourney00.github.io/IM-HOI.github.io/) ha sido aceptado en CVPR 2024.
- ***04 de Enero, 2024:*** 🔈🔈 ¡Completa [el formulario](https://forms.gle/3MDh3b4szhFwcYa26) para obtener acceso a IMHD$`^2`$!

## Contenido
- [Características del conjunto de datos](#dataset-features)
- [Estructura del conjunto de datos](#dataset-structure)
- [Primeros pasos](#getting-started)
  - [Para Windows](#for-windows)
  - [Para Ubuntu](#for-ubuntu)
  - [Cómo usarlo](#how-to-use)
- [Preguntas frecuentes](#faqs)

## Características del conjunto de datos
IMHD$`^2`$ se caracteriza por:
- Anotación de movimiento humano en formato SMPL-H, basada en [EasyMocap](https://github.com/zju3dv/EasyMocap/tree/master)
- Anotación de movimiento de objetos, basada en [PHOSA](https://github.com/facebookresearch/phosa)
- Geometría de objetos bien escaneada, usando [Polycam](https://poly.cam/)
- Medición de sensores IMU montados en objetos, usando [Movella DOT](https://www.movella.com/products/wearables/movella-dot)
- Videos RGB de 32 vistas y segmentaciones a nivel de instancia, basados en [SAM](https://github.com/facebookresearch/segment-anything), [Track-Anything](https://github.com/gaomingqi/Track-Anything) y [XMem](https://github.com/hkchengrex/XMem)
- Detección de puntos clave humanos 2D y 3D de 32 vistas, usando [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) y [MediaPipe](https://github.com/google/mediapipe)

## Estructura del conjunto de datos
```
data/
|--calibrations/           # intrínsecas de cámara y extrínsecas mundo-a-cámara
|--object_templates/       # geometría sin procesar y submuestreada
|--imu_preprocessed/       # señal IMU preprocesada
|--keypoints2d/            # puntos clave corporales en formato OP25 y puntos clave manuales en formato MediaPipe
|--keypoints3d/            # puntos clave corporales en formato OP25 y puntos clave manuales en formato MediaPipe
|--video_release/          # videos sin procesar desde 32 vistas múltiples
|--mask_release/           # segmentaciones separadas de humano y objeto desde 32 vistas múltiples
|--ground_truth/           # movimiento humano en formato SMPL-H y movimiento de objeto rígido
|----<date>/
|------<segment_name>/
|--------<sequence_name>/
|----------gt_<part_id>_<start>_<end>.pkl
```
Todos los subdirectorios tienen una estructura detallada similar a la mostrada para el ground truth. En particular, dado que las anotaciones de movimiento de algunas partes en ciertas secuencias no son ideales, puede haber varios archivos `.pkl` en una carpeta de secuencia. Para interpretar el significado del nombre de archivo de los archivos `.pkl` finales, aquí hay un ejemplo: `gt_0_10_100.pkl: la primera parte de movimiento que comienza en el frame_10 y termina en el frame_100`.

## Primeros pasos
Probamos nuestro código en ``Windows 10``, ``Windows 11``, ``Ubuntu 18.04 LTS`` y ``Ubuntu 20.04 LTS``.

Todas las dependencias:
> python>=3.8  
> CUDA=11.7  
> torch=1.13.0  
> pytorch3d  
> opencv-python  
> matplotlib  
> smplx

### Para Windows
```
conda create -n imhd2 python=3.8 -y
conda activate imhd2
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e . --ignore-installed PyYAML
```

### Para Ubuntu
```
conda create -n imhd2 python=3.8 -y
conda activate imhd2
conda install --file conda_install_cuda117_pakage.txt -c nvidia
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

### Cómo usarlo
1. Prepara los datos. Descarga IMHD$`^2`$ desde [aquí](https://forms.gle/3MDh3b4szhFwcYa26) y colócalo en el directorio raíz según la [estructura predefinida](#dataset-structure).
2. Prepara el modelo corporal. Descarga [SMPL-H](https://mano.is.tue.mpg.de/login.php) (el modelo extendido SMPL+H) y coloca los archivos del modelo en la carpeta `body_model/`. En general, la estructura de la carpeta `body_model/` debe ser:
```
body_model/
|--__init__.py
|--body_model.py
|--utils.py
|--smplh/
|----info.txt
|----LICENSE.txt
|----female/
|------model.npz
|----male/
|------model.npz
|----neutral/
|------model.npz
```
3. Ejecuta `python visualization.py` para verificar cómo cargar y visualizar IMHD$`^2`$. Los resultados se guardarán en `visualizations/`.

## Preguntas frecuentes
**P1: ¿En qué coordenadas están los movimientos de ground truth? ¿Cómo alinear todos los movimientos entre diferentes fechas?**

*R1: Los movimientos de ground truth están en las **coordenadas del mundo**, que se calibraron usando un sistema de múltiples cámaras y pueden variar entre fechas. Para alinearlos, puedes usar los parámetros de cámara proporcionados en ``calibrations/`` para transformar todos los datos de movimiento a coordenadas de cámara.*

**P2: ¿Con qué categoría de objeto interactúan los movimientos denominados 'bat' en ``20230825/`` y ``20230827/``?**

*R2: La categoría de objeto interactuante de los movimientos en ``20230825/`` y ``20230827/`` es bate de béisbol, correspondiente a **'baseball'** en la carpeta ``object_templates/``.*

**P3: ¿Qué cámara sirve como vista principal?**

*R3: La vista principal proviene de la cámara etiquetada como '1'(comenzando desde 0).*

**P3: ¿Cómo decodificar los videos sin procesar a imágenes?**

*R3: Por favor, usa el comando: `ffmpeg -i <input_path> -qscale:v 2 -f image2 -v error -start_number 0 -threads 64 output/%06d.jpg`*

**P4: Para la evaluación en el conjunto de datos IMHD, ¿qué vista de cámara y fps de video debo usar?**

*R4: Usa la segunda vista de la lista de cámaras (data2.mp4 para video RGB y 1.mp4 para video de máscara). Y las métricas del conjunto de datos IMHD reportadas en nuestro artículo (incluyendo las líneas base) se evalúan en datos de 60fps.*

## Citación
Si consideras que nuestros datos o artículo son útiles, por favor considera citarlos:
```bibtex
@inproceedings{zhao2024imhoi,
  title={I'm hoi: Inertia-aware monocular capture of 3d human-object interactions},
  author={Zhao, Chengfeng and Zhang, Juze and Du, Jiashen and Shan, Ziwei and Wang, Junye and Yu, Jingyi and Wang, Jingya and Xu, Lan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={729--741},
  year={2024}
}
```

## Agradecimientos
Este trabajo fue apoyado por el Programa Nacional Clave de I\&D de China (2022YFF0902301), el Programa de Desarrollo de Capacidad de Colegios Locales de Shanghái (22010502800). También reconocemos el apoyo del Centro de Ciencias de Fronteras de Shanghái para la Inteligencia Artificial Centrada en el Humano (ShangHAI).

Agradecemos a Jingyan Zhang y Hongdi Yang por configurar el sistema de captura. Agradecemos a Jingyan Zhang, Zining Song, Jierui Xu, Weizhi Wang, Gubin Hu, Yelin Wang, Zhiming Yu, Xuanchen Liang, af y zr por la recopilación de datos. Agradecemos a Xiao Yu, Yuntong Liu y Xiaofan Gu por la verificación y anotación de datos.

## Licencias
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a><br />Este trabajo está licenciado bajo una <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Licencia Creative Commons Atribución-NoComercial-CompartirIgual 4.0 Internacional</a>.
