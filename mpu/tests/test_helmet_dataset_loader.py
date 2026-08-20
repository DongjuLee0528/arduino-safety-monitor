"""
Unit tests for the corrected HelmetDataset loader (per-object crop semantics).

Baseline training contract:
  person_with_helmet → label 1 (helmet)
  person_no_helmet   → label 0 (no_helmet)
  helmet / head_with_helmet / head / face / unknown → SKIPPED

These tests verify the loader contract WITHOUT the real training dataset by
using in-memory XML fixtures and temporary synthetic images.

Verified behaviours (16 contract tests):
1.  person_with_helmet → label 1
2.  person_no_helmet   → label 0
3.  helmet object      → skipped
4.  head               → skipped
5.  head_with_helmet   → skipped
6.  face               → skipped
7.  unknown annotation → skipped
8.  mixed image (person_with_helmet + person_no_helmet) → two independent samples
9.  bbox clipping (out-of-bound coords → clipped to image boundary)
10. bbox < MIN_DIM (32 px) → skipped
11. malformed bbox (xmin >= xmax) → skipped
12. malformed XML → safe empty list, NOT a default no_helmet label
13. grouped train/val split — source image never crosses split
14. no source-image leakage (train ∩ val sources = ∅)
15. class mapping contract: 0 = no_helmet, 1 = helmet
16. Dataset B annotations (hat/person) not silently mapped
"""

import os
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from mpu.ai.dataset.loader import HelmetDataset, _split_source_groups


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_xml(path, objects):
    """Write a minimal Pascal VOC XML."""
    lines = ['<annotation>', '  <folder>test</folder>']
    for obj in objects:
        lines.append('  <object>')
        if 'name' in obj:
            lines.append(f'    <name>{obj["name"]}</name>')
        if 'bndbox' in obj:
            bb = obj['bndbox']
            lines.append('    <bndbox>')
            for k in ('xmin', 'ymin', 'xmax', 'ymax'):
                if k in bb:
                    lines.append(f'      <{k}>{bb[k]}</{k}>')
            lines.append('    </bndbox>')
        elif 'xmin' in obj:
            lines.append('    <bndbox>')
            for k in ('xmin', 'ymin', 'xmax', 'ymax'):
                lines.append(f'      <{k}>{obj[k]}</{k}>')
            lines.append('    </bndbox>')
        lines.append('  </object>')
    lines.append('</annotation>')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def _make_image(path, w=640, h=480):
    img = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
    img.save(path)


def _fresh_ds():
    """Return a HelmetDataset instance with correct baseline label sets (no __init__)."""
    ds = HelmetDataset.__new__(HelmetDataset)
    ds.helmet_labels = {"person_with_helmet"}
    ds.no_helmet_labels = {"person_no_helmet"}
    return ds


# ---------------------------------------------------------------------------
# 1-7, 11-12: _parse_xml_objects unit tests
# ---------------------------------------------------------------------------

class TestParseXmlObjects(unittest.TestCase):

    def setUp(self):
        self.ds = _fresh_ds()
        self.tmpdir = tempfile.mkdtemp()

    def _xml(self, filename, objects):
        path = os.path.join(self.tmpdir, filename)
        _write_xml(path, objects)
        return path

    # Contract 1
    def test_person_with_helmet_maps_to_label_1(self):
        path = self._xml('pwh.xml', [{'name': 'person_with_helmet',
                                       'xmin': 10, 'ymin': 20, 'xmax': 200, 'ymax': 400}])
        result = self.ds._parse_xml_objects(path)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], 1)

    # Contract 2
    def test_person_no_helmet_maps_to_label_0(self):
        path = self._xml('pnh.xml', [{'name': 'person_no_helmet',
                                       'xmin': 5, 'ymin': 10, 'xmax': 180, 'ymax': 380}])
        result = self.ds._parse_xml_objects(path)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], 0)

    # Contract 3
    def test_helmet_object_crop_is_skipped(self):
        path = self._xml('helmet.xml', [{'name': 'helmet',
                                          'xmin': 10, 'ymin': 20, 'xmax': 80, 'ymax': 60}])
        self.assertEqual(self.ds._parse_xml_objects(path), [])

    # Contract 4
    def test_head_is_skipped(self):
        path = self._xml('head.xml', [{'name': 'head',
                                        'xmin': 5, 'ymin': 10, 'xmax': 60, 'ymax': 80}])
        self.assertEqual(self.ds._parse_xml_objects(path), [])

    # Contract 5
    def test_head_with_helmet_is_skipped(self):
        path = self._xml('hwh.xml', [{'name': 'head_with_helmet',
                                       'xmin': 0, 'ymin': 0, 'xmax': 50, 'ymax': 60}])
        self.assertEqual(self.ds._parse_xml_objects(path), [])

    # Contract 6
    def test_face_is_skipped(self):
        path = self._xml('face.xml', [{'name': 'face',
                                        'xmin': 0, 'ymin': 0, 'xmax': 30, 'ymax': 40}])
        self.assertEqual(self.ds._parse_xml_objects(path), [])

    # Contract 7
    def test_unknown_annotation_is_skipped(self):
        path = self._xml('unknown.xml', [{'name': 'car',
                                           'xmin': 0, 'ymin': 0, 'xmax': 100, 'ymax': 100}])
        self.assertEqual(self.ds._parse_xml_objects(path), [])

    # Contract 11
    def test_degenerate_bbox_xmin_equals_xmax_is_skipped(self):
        path = self._xml('degen_x.xml', [{'name': 'person_with_helmet',
                                           'xmin': 50, 'ymin': 10, 'xmax': 50, 'ymax': 300}])
        self.assertEqual(self.ds._parse_xml_objects(path), [])

    def test_degenerate_bbox_ymin_greater_than_ymax_is_skipped(self):
        path = self._xml('degen_y.xml', [{'name': 'person_no_helmet',
                                           'xmin': 10, 'ymin': 200, 'xmax': 200, 'ymax': 100}])
        self.assertEqual(self.ds._parse_xml_objects(path), [])

    def test_missing_bndbox_skips_object(self):
        path = os.path.join(self.tmpdir, 'no_bndbox.xml')
        with open(path, 'w') as f:
            f.write('<annotation><object><name>person_with_helmet</name></object></annotation>')
        self.assertEqual(self.ds._parse_xml_objects(path), [])

    # Contract 12
    def test_xml_parse_failure_returns_empty_list_not_no_helmet(self):
        path = os.path.join(self.tmpdir, 'broken.xml')
        with open(path, 'w') as f:
            f.write('<annotation><unclosed>')
        self.assertEqual(self.ds._parse_xml_objects(path), [])

    def test_bbox_stored_as_xywh(self):
        path = self._xml('xywh.xml', [{'name': 'person_with_helmet',
                                        'xmin': 10, 'ymin': 20, 'xmax': 210, 'ymax': 420}])
        (x, y, w, h), label = self.ds._parse_xml_objects(path)[0]
        self.assertEqual((x, y, w, h), (10, 20, 200, 400))

    # Contract 16: Dataset B annotation names must not silently map
    def test_dataset_b_hat_annotation_is_skipped(self):
        path = self._xml('hat.xml', [{'name': 'hat',
                                       'xmin': 0, 'ymin': 0, 'xmax': 100, 'ymax': 100}])
        self.assertEqual(self.ds._parse_xml_objects(path), [])

    def test_dataset_b_person_annotation_is_skipped(self):
        path = self._xml('person.xml', [{'name': 'person',
                                          'xmin': 0, 'ymin': 0, 'xmax': 100, 'ymax': 300}])
        self.assertEqual(self.ds._parse_xml_objects(path), [])


# ---------------------------------------------------------------------------
# Contract 8: mixed image → independent samples
# ---------------------------------------------------------------------------

class TestMixedImageAnnotations(unittest.TestCase):

    def setUp(self):
        self.ds = _fresh_ds()
        self.tmpdir = tempfile.mkdtemp()

    def test_mixed_image_produces_two_independent_samples(self):
        path = os.path.join(self.tmpdir, 'mixed.xml')
        _write_xml(path, [
            {'name': 'person_with_helmet', 'xmin': 10, 'ymin': 20, 'xmax': 200, 'ymax': 400},
            {'name': 'person_no_helmet',   'xmin': 250, 'ymin': 20, 'xmax': 440, 'ymax': 400},
        ])
        result = self.ds._parse_xml_objects(path)
        self.assertEqual(len(result), 2)
        labels = {label for _, label in result}
        self.assertIn(0, labels)
        self.assertIn(1, labels)

    def test_mixed_image_does_not_include_non_person_annotations(self):
        """helmet/head/face in same image must not produce extra samples."""
        path = os.path.join(self.tmpdir, 'mixed_full.xml')
        _write_xml(path, [
            {'name': 'person_with_helmet', 'xmin': 10, 'ymin': 20, 'xmax': 200, 'ymax': 400},
            {'name': 'helmet',             'xmin': 10, 'ymin': 20, 'xmax': 50,  'ymax': 50},
            {'name': 'head_with_helmet',   'xmin': 20, 'ymin': 20, 'xmax': 60,  'ymax': 70},
            {'name': 'person_no_helmet',   'xmin': 250, 'ymin': 20, 'xmax': 440, 'ymax': 400},
            {'name': 'head',               'xmin': 260, 'ymin': 20, 'xmax': 290, 'ymax': 60},
            {'name': 'face',               'xmin': 262, 'ymin': 30, 'xmax': 285, 'ymax': 55},
        ])
        result = self.ds._parse_xml_objects(path)
        self.assertEqual(len(result), 2)


# ---------------------------------------------------------------------------
# Contract 9: bbox clipping   Contract 10: min_dim=32
# ---------------------------------------------------------------------------

class TestBboxCliping(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.annot_dir = os.path.join(self.tmpdir, 'Annotations')
        self.img_dir   = os.path.join(self.tmpdir, 'Images')
        os.makedirs(self.annot_dir)
        os.makedirs(self.img_dir)

    def _load(self, xml_name, objects, img_w=640, img_h=480):
        img_path = os.path.join(self.img_dir, xml_name.replace('.xml', '.png'))
        _make_image(img_path, w=img_w, h=img_h)
        _write_xml(os.path.join(self.annot_dir, xml_name), objects)
        ds = HelmetDataset.__new__(HelmetDataset)
        ds.helmet_labels = {"person_with_helmet"}
        ds.no_helmet_labels = {"person_no_helmet"}
        return ds._load_dataset(self.annot_dir, self.img_dir, '.png', 'TEST')

    # Contract 9
    def test_out_of_bound_bbox_clipped_to_image_boundary(self):
        samples = self._load('clip.xml', [
            {'name': 'person_with_helmet', 'xmin': -10, 'ymin': -5, 'xmax': 700, 'ymax': 500}
        ], img_w=100, img_h=80)
        self.assertEqual(len(samples), 1)
        _, (x, y, w, h), label = samples[0]
        self.assertEqual((x, y, w, h), (0, 0, 100, 80))
        self.assertEqual(label, 1)

    # Contract 10
    def test_small_bbox_under_min_dim_is_skipped(self):
        samples = self._load('small.xml', [
            {'name': 'person_with_helmet', 'xmin': 0, 'ymin': 0, 'xmax': 31, 'ymax': 300}
        ])
        self.assertEqual(samples, [])

    def test_bbox_exactly_at_min_dim_is_kept(self):
        samples = self._load('exact.xml', [
            {'name': 'person_with_helmet', 'xmin': 0, 'ymin': 0, 'xmax': 32, 'ymax': 32}
        ])
        self.assertEqual(len(samples), 1)

    def test_small_height_under_min_dim_is_skipped(self):
        samples = self._load('small_h.xml', [
            {'name': 'person_no_helmet', 'xmin': 0, 'ymin': 0, 'xmax': 200, 'ymax': 31}
        ])
        self.assertEqual(samples, [])


# ---------------------------------------------------------------------------
# __getitem__ integration tests
# ---------------------------------------------------------------------------

class TestHelmetDatasetGetItem(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.img_path = os.path.join(self.tmpdir, 'img0.png')
        _make_image(self.img_path, w=640, h=480)

        ds = HelmetDataset.__new__(HelmetDataset)
        ds.helmet_labels = {"person_with_helmet"}
        ds.no_helmet_labels = {"person_no_helmet"}
        from torchvision import transforms
        ds.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        ds.samples = [
            (self.img_path, (10, 20, 200, 350), 1),
            (self.img_path, (250, 20, 200, 350), 0),
        ]
        self.ds = ds

    def test_getitem_tensor_shape_is_3_224_224(self):
        img, _ = self.ds[0]
        self.assertEqual(img.shape, (3, 224, 224))

    def test_getitem_label_helmet_is_1(self):
        _, label = self.ds[0]
        self.assertEqual(label.item(), 1)

    def test_getitem_label_no_helmet_is_0(self):
        _, label = self.ds[1]
        self.assertEqual(label.item(), 0)

    def test_getitem_tensor_dtype_is_float32(self):
        img, _ = self.ds[0]
        self.assertEqual(img.dtype, torch.float32)

    def test_empty_bbox_raises_runtime_error(self):
        self.ds.samples = [(self.img_path, (1000, 1000, 10, 10), 1)]
        with self.assertRaises(RuntimeError):
            self.ds[0]


# ---------------------------------------------------------------------------
# Contract 15: class mapping constant
# ---------------------------------------------------------------------------

class TestClassMappingContract(unittest.TestCase):

    def test_annotation_labels_no_helmet_is_0(self):
        self.assertEqual(HelmetDataset.ANNOTATION_LABELS["person_no_helmet"], 0)

    def test_annotation_labels_helmet_is_1(self):
        self.assertEqual(HelmetDataset.ANNOTATION_LABELS["person_with_helmet"], 1)

    def test_only_two_labels_in_annotation_labels(self):
        self.assertEqual(set(HelmetDataset.ANNOTATION_LABELS.keys()),
                         {"person_no_helmet", "person_with_helmet"})


# ---------------------------------------------------------------------------
# class distribution helper
# ---------------------------------------------------------------------------

class TestClassDistribution(unittest.TestCase):

    def test_distribution_counts_correct_labels(self):
        ds = HelmetDataset.__new__(HelmetDataset)
        ds.samples = [
            ('a.jpg', (0, 0, 100, 200), 1),
            ('b.jpg', (0, 0, 100, 200), 0),
            ('c.jpg', (0, 0, 100, 200), 1),
        ]
        dist = ds.get_class_distribution()
        self.assertEqual(dist['helmet'], 2)
        self.assertEqual(dist['no_helmet'], 1)
        self.assertEqual(dist['total'], 3)


# ---------------------------------------------------------------------------
# Contracts 13 & 14: grouped train/val split, no leakage
# ---------------------------------------------------------------------------

class TestSourceGroupedSplit(unittest.TestCase):

    def test_source_image_never_crosses_train_and_val(self):
        samples = [
            ('a.jpg', (0, 0, 100, 200), 1),
            ('a.jpg', (150, 0, 100, 200), 0),
            ('b.jpg', (0, 0, 100, 200), 1),
            ('c.jpg', (0, 0, 100, 200), 0),
        ]
        train_idx, val_idx = _split_source_groups(samples, 2 / 3)
        train_sources = {samples[i][0] for i in train_idx}
        val_sources   = {samples[i][0] for i in val_idx}
        # Contract 13 & 14: disjoint
        self.assertTrue(train_sources.isdisjoint(val_sources))
        # All indices covered
        self.assertEqual(sorted(train_idx + val_idx), list(range(len(samples))))

    def test_split_is_deterministic_with_fixed_seed(self):
        samples = [('img%d.jpg' % i, (0, 0, 100, 200), i % 2) for i in range(20)]
        t1, v1 = _split_source_groups(samples, 0.8)
        t2, v2 = _split_source_groups(samples, 0.8)
        self.assertEqual(t1, t2)
        self.assertEqual(v1, v2)


if __name__ == "__main__":
    unittest.main()
