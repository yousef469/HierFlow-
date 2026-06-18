"""Real Image Dataset: Stanford Dogs → (688-dim tensor, 64×64 image) pairs.

Downloads Stanford Dogs (~700MB), maps breed names to KB wnids,
and produces paired training data.

Stanford Dogs: http://vision.stanford.edu/aditya86/ImageNetDogs/
"""

import os
import sys
import json
import tarfile
import urllib.request
import numpy as np
from pathlib import Path

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from frontend.encoder import prompt_to_tensor
from frontend import knowledge_base as kb


STANFORD_DOGS_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
ANNOTATIONS_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar"
DATA_DIR = os.path.join(_pkg_root, "data", "stanford_dogs")
IMAGES_DIR = os.path.join(DATA_DIR, "Images")
ANNOT_DIR = os.path.join(DATA_DIR, "Annotation")


_BREED_TO_WNID = {}
for wid, syn in kb.SYNSETS.items():
    name = syn["name"].lower().replace("_", " ")
    _BREED_TO_WNID[name] = wid
    # Also map plural
    _BREED_TO_WNID[name + "s"] = wid

# Add Stanford Dog breed names to KB wnid mapping
_STANFORD_BREED_OVERRIDES = {
    "n02085620": 2084071,  # Chihuahua → dog
    "n02085782": 2084071,  # Japanese spaniel → dog
    "n02085936": 2084071,  # Maltese → dog
    "n02086079": 2084071,  # Pekinese → dog
    "n02086240": 2084071,  # Shih-Tzu → dog
    "n02086646": 2084071,  # Blenheim spaniel → dog
    "n02086910": 2084071,  # papillon → dog
    "n02087046": 2084071,  # toy terrier → dog
    "n02087394": 2084071,  # Rhodesian ridgeback → dog
    "n02088094": 2084071,  # Afghan hound → dog
    "n02088238": 2084071,  # basset → dog
    "n02088364": 2084071,  # beagle → dog
    "n02088466": 2084071,  # bloodhound → dog
    "n02088632": 2084071,  # bluetick → dog
    "n02089078": 2084071,  # black-and-tan coonhound → dog
    "n02089867": 2084071,  # Walker hound → dog
    "n02089973": 2084071,  # English foxhound → dog
    "n02090379": 2084071,  # redbone → dog
    "n02090622": 2084071,  # borzoi → dog
    "n02090721": 2084071,  # Irish wolfhound → dog
    "n02091032": 2084071,  # Italian greyhound → dog
    "n02091134": 2084071,  # whippet → dog
    "n02091244": 2084071,  # Ibizan hound → dog
    "n02091467": 2084071,  # Norwegian elkhound → dog
    "n02091635": 2084071,  # Otterhound → dog
    "n02091831": 2084071,  # Saluki → dog
    "n02092002": 2084071,  # Scottish deerhound → dog
    "n02092339": 2084071,  # Weimaraner → dog
    "n02092474": 2084071,  # Staffordshire bullterrier → dog
    "n02093256": 2084071,  # American Staffordshire terrier → dog
    "n02093428": 2084071,  # Bedlington terrier → dog
    "n02093647": 2084071,  # Border terrier → dog
    "n02093754": 2084071,  # Kerry blue terrier → dog
    "n02093859": 2084071,  # Irish terrier → dog
    "n02093991": 2084071,  # Norfolk terrier → dog
    "n02094114": 2084071,  # Norwich terrier → dog
    "n02094258": 2084071,  # Yorkshire terrier → dog
    "n02094433": 2084071,  # wire-haired fox terrier → dog
    "n02095314": 2084071,  # Lakeland terrier → dog
    "n02095570": 2084071,  # Sealyham terrier → dog
    "n02095889": 2084071,  # Airedale → dog
    "n02096051": 2084071,  # cairn → dog
    "n02096177": 2084071,  # Australian terrier → dog
    "n02096294": 2084071,  # Dandie Dinmont → dog
    "n02096437": 2084071,  # Boston bull → dog
    "n02096585": 2084071,  # miniature schnauzer → dog
    "n02097047": 2084071,  # giant schnauzer → dog
    "n02097130": 2084071,  # standard schnauzer → dog
    "n02097209": 2084071,  # Scottish terrier → dog
    "n02097298": 2084071,  # Tibetan terrier → dog
    "n02097474": 2084071,  # Australian silky terrier → dog
    "n02097658": 2084071,  # soft-coated wheaten terrier → dog
    "n02098105": 2084071,  # West Highland white terrier → dog
    "n02098286": 2084071,  # Lhasa → dog
    "n02098413": 2084071,  # flat-coated retriever → dog
    "n02099267": 2084071,  # curly-coated retriever → dog
    "n02099429": 2084071,  # golden retriever → 2084074
    "n02099601": 2084074,  # golden retriever → 2084074
    "n02099712": 2084071,  # Labrador retriever → dog
    "n02099849": 2084071,  # Chesapeake Bay retriever → dog
    "n02100236": 2084071,  # German short-haired pointer → dog
    "n02100583": 2084071,  # vizsla → dog
    "n02100735": 2084071,  # English setter → dog
    "n02100877": 2084071,  # Irish setter → dog
    "n02101006": 2084071,  # Gordon setter → dog
    "n02101388": 2084071,  # Brittany spaniel → dog
    "n02101556": 2084071,  # clumber → dog
    "n02102040": 2084071,  # English springer → dog
    "n02102177": 2084071,  # Welsh springer spaniel → dog
    "n02102318": 2084071,  # cocker spaniel → dog
    "n02102480": 2084071,  # Sussex spaniel → dog
    "n02102973": 2084071,  # Irish water spaniel → dog
    "n02104029": 2084071,  # kuvasz → dog
    "n02104365": 2084071,  # schipperke → dog
    "n02105056": 2084071,  # groenendael → dog
    "n02105162": 2084071,  # malinois → dog
    "n02105251": 2084071,  # briard → dog
    "n02105412": 2084071,  # kelpie → dog
    "n02105505": 2084071,  # komondor → dog
    "n02105641": 2084071,  # Old English sheepdog → dog
    "n02105855": 2084071,  # Shetland sheepdog → dog
    "n02106030": 2084071,  # collie → dog
    "n02106166": 2084071,  # Border collie → dog
    "n02106382": 2084071,  # Bouvier des Flandres → dog
    "n02106550": 2084071,  # Rottweiler → dog
    "n02106662": 2084071,  # German shepherd → dog
    "n02107142": 2084071,  # Doberman → dog
    "n02107312": 2084071,  # miniature pinscher → dog
    "n02107574": 2084071,  # Greater Swiss Mountain dog → dog
    "n02107683": 2084071,  # Bernese mountain dog → dog
    "n02107908": 2084071,  # Appenzeller → dog
    "n02108000": 2084071,  # EntleBucher → dog
    "n02108089": 2084072,  # boxer → husky
    "n02108422": 2084071,  # bull mastiff → dog
    "n02108551": 2084071,  # Tibetan mastiff → dog
    "n02108915": 2084071,  # French bulldog → dog
    "n02109047": 2084075,  # Great Dane → bulldog
    "n02109525": 2084071,  # Saint Bernard → dog
    "n02109961": 2084071,  # Eskimo dog → husky
    "n02110063": 2084071,  # malamute → husky
    "n02110185": 2084072,  # Siberian husky → 2084072
    "n02110627": 2084071,  # affenpinscher → dog
    "n02110806": 2084071,  # basenji → dog
    "n02110958": 2084071,  # pug → 2084076
    "n02111129": 2084071,  # Leonberg → dog
    "n02111277": 2084071,  # Newfoundland → dog
    "n02111500": 2084071,  # Great Pyrenees → dog
    "n02111889": 2084071,  # Samoyed → dog
    "n02112018": 2084071,  # Pomeranian → dog
    "n02112137": 2084071,  # chow → dog
    "n02112350": 2084071,  # keeshond → dog
    "n02112706": 2084071,  # Brabancon griffon → dog
    "n02113023": 2084071,  # Pembroke → dog
    "n02113186": 2084071,  # Cardigan → dog
    "n02113624": 2084071,  # toy poodle → dog
    "n02113712": 2084073,  # miniature poodle → 2084073
    "n02113799": 2084073,  # standard poodle → 2084073
    "n02113978": 2084071,  # Mexican hairless → dog
}


def _download_and_extract(url, target_dir):
    """Download and extract a tar file."""
    os.makedirs(target_dir, exist_ok=True)
    fname = os.path.basename(url)
    local_path = os.path.join(target_dir, fname)

    if not os.path.exists(local_path):
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, local_path)
        print("Done.")

    extract_dir = os.path.join(target_dir, fname.replace(".tar", ""))
    if not os.path.exists(extract_dir):
        print(f"Extracting {fname}...")
        with tarfile.open(local_path) as tar:
            tar.extractall(path=target_dir)
        print("Done.")

    return extract_dir


def _load_annotation(annotation_path):
    """Load bounding box from Stanford Dogs annotation XML.

    Returns (x, y, w, h) in relative [0,1] coordinates, or None.
    """
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        obj = root.find("object")
        if obj is not None:
            bndbox = obj.find("bndbox")
            if bndbox is not None:
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)
                # Get image size
                size = root.find("size")
                if size is not None:
                    img_w = int(size.find("width").text)
                    img_h = int(size.find("height").text)
                    cx = (xmin + xmax) / 2 / img_w
                    cy = (ymin + ymax) / 2 / img_h
                    w = (xmax - xmin) / img_w
                    h = (ymax - ymin) / img_h
                    return (cx, cy, w, h)
    except Exception:
        pass
    return None


def _prompt_for_dog(breed_name):
    """Generate a prompt for a dog image."""
    color = np.random.choice([
        "brown", "black", "white", "golden", "gray", "red", "tan"
    ])
    action = np.random.choice([
        "standing", "sitting", "running", "lying", "walking"
    ])
    location = np.random.choice([
        "grass", "field", "snow", "ground", "road", "rocks"
    ])
    style = np.random.choice([
        "",
        " on " + location,
    ])
    return f"a {color} {breed_name} {action}{style}"


class StanfordDogsDataset(Dataset):
    """Stanford Dogs → (688-dim tensor, resized image).

    Each item: (cond_tensor, image_tensor)
        cond_tensor: (688,) float — from frontend
        image_tensor: (3, H, W) float in [0, 1] — resized image
    """

    def __init__(self, img_size=64, download=True):
        self.img_size = img_size
        self.pairs = []

        # Verify/collect data
        if not os.path.exists(IMAGES_DIR):
            if download:
                print("Downloading Stanford Dogs...")
                _download_and_extract(STANFORD_DOGS_URL, DATA_DIR)
                _download_and_extract(ANNOTATIONS_URL, DATA_DIR)
            else:
                raise FileNotFoundError(f"Stanford Dogs not found at {IMAGES_DIR}")

        # Walk breed directories
        breed_dirs = sorted(os.listdir(IMAGES_DIR))
        for breed_dir in breed_dirs:
            breed_path = os.path.join(IMAGES_DIR, breed_dir)
            if not os.path.isdir(breed_path):
                continue

            # Extract ImageNet synset ID (e.g., "n02085620" from "n02085620-Chihuahua")
            breed_id = breed_dir.split("-")[0] if "-" in breed_dir else breed_dir
            wnid = _STANFORD_BREED_OVERRIDES.get(breed_id, 2084071)  # default to dog

            # Breed name
            breed_name = breed_dir.split("-", 1)[1] if "-" in breed_dir else breed_dir
            breed_name = breed_name.replace("_", " ").lower()

            # Annotation dir
            annot_breed_dir = os.path.join(ANNOT_DIR, breed_dir)
            if not os.path.exists(annot_breed_dir):
                annot_breed_dir = None

            # Collect image files
            img_extensions = {".jpg", ".jpeg", ".png"}
            images = sorted([f for f in os.listdir(breed_path)
                             if os.path.splitext(f)[1].lower() in img_extensions])

            for img_name in images:
                img_path = os.path.join(breed_path, img_name)

                # Try to load annotation for bounding box
                bbox = None
                if annot_breed_dir is not None:
                    annot_name = os.path.splitext(img_name)[0]
                    annot_path = os.path.join(annot_breed_dir, annot_name)
                    if os.path.exists(annot_path):
                        bbox = _load_annotation(annot_path)

                self.pairs.append({
                    "img_path": img_path,
                    "wnid": wnid,
                    "breed_name": breed_name,
                    "bbox": bbox,
                })

        print(f"Loaded {len(self.pairs)} images from {len(breed_dirs)} breeds")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]

        # Load image
        img = Image.open(item["img_path"]).convert("RGB")
        orig_w, orig_h = img.size

        # Build prompt
        prompt = _prompt_for_dog(item["breed_name"])

        # Generate conditioning tensor from prompt
        cond, layout = prompt_to_tensor(prompt)

        # If we have bbox from annotation, override the layout positions
        bbox = item.get("bbox")
        if bbox is not None and len(layout["objects"]) > 0:
            cx, cy, bw, bh = bbox
            # Update the first real object (skip bg objects like sky/ground)
            for obj in layout["objects"]:
                syn = kb.SYNSETS.get(obj["wnid"], {})
                name = syn.get("name", "")
                if name not in ("sky", "ground", "grass", "tree", "water", "cloud", "mountain", "sun"):
                    obj["x"] = min(1.0, max(0.0, cx))
                    obj["y"] = min(1.0, max(0.0, cy))
                    obj["w"] = min(1.0, max(0.05, bw))
                    obj["h"] = min(1.0, max(0.05, bh))
                    break

            # Re-encode with updated layout
            from frontend.encoder import encode_scene_to_tensor
            try:
                cond = encode_scene_to_tensor(layout)
            except Exception:
                pass

        # Resize image
        img_resized = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_t = torch.from_numpy(np.array(img_resized)).float().permute(2, 0, 1) / 255.0

        return cond.float(), img_t
