from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from typing import List
from algorithms.valid_print import isPrintable
from io import BytesIO
from keras.models import load_model
import numpy as np
from algorithms.smiley_gan.utils import finalize_generated_img, downsize
from PIL import Image

class Grid(BaseModel):
    cells: List[List[int]]


gan_smiley = load_model('algorithms/smiley_gan/saved_models/smiley_gan_h5')
generator_smiley = gan_smiley.get_layer('model_128')
discriminator_smiley = gan_smiley.get_layer('model_127')
app = FastAPI()


@app.post("/isSmileyFace")
async def root(uploaded_image: UploadFile = File(media_type='image/jpeg')):
    content = BytesIO(await uploaded_image.read())
    img = Image.open(content)
    img = downsize(np.array(img))
    similarity = discriminator_smiley.predict(np.array([img]))[0][0]
    return {"similarity": float(similarity)}


@app.post("/generateSmiley")
async def root():
    image = BytesIO()
    noise = np.random.normal(0, 1, (1, 100))
    gen_imgs = generator_smiley.predict(noise)
    print(gen_imgs.shape)
    smiley = finalize_generated_img(gen_imgs[0])
    smiley_image = Image.fromarray(smiley)
    smiley_image.save(image, format='JPEG', quality=100)
    image.seek(0)
    return Response(content=image.read(), media_type='image/jpeg')
