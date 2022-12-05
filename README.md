# Dataset augmentation with streamlit webui

## Dependencies

```
python3 -m venv <insert virtual name>
source <virtual name>/bin/activate

pip install streamlit
pip install opencv
```

## How to run?

```
streamlit run 1_About_st.data-aug.py
```

## Features
**<details><summary>Grayscale images</summary>**
1. Input image file path
2. Input image destination path
3. Click on 'Grayscale Images'
4. View grayscaled images with slider
</details>

**<details><summary>Resize images</summary>**
1. Input image file path
2. Input image destination path
3. Insert image width
4. Insert images height
5. Click on 'Resize Images'
6. View resized images with slider
</details>

**<details><summary>Overlay images</summary>**
1. Input background image
2. Input overlaying image
3. Insert top left x coordinates of location
4. Insert top left y coordinates of location
5. Click on 'Overlay Images'
6. View overlayed images with slider
</details>