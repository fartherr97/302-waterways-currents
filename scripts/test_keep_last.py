"""keep_last.py against a fake published site (no network)"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import keep_last as kl  # noqa: E402

SITE = 'https://example.test/site/'
MAN = {'updated': '2026-11-25T00:36:00+00:00', 'tiles': ['t5_3', 't6_3']}
PUBLISHED = {
    'watertemp/manifest.json': json.dumps(MAN).encode(),
    'watertemp/o.png': b'overview', 'watertemp/t5_3.png': b'tile53', 'watertemp/t6_3.png': b'tile63',
    'watertemp/stations.json': b'{"rows": []}',
    'inlets.json': b'{"updated": "2026-11-25T00:35:34+00:00", "cells": []}',
}


def fake(files):
    def f(url, timeout=60):
        k = url[len(SITE):]
        if k not in files:
            raise OSError(f'404 {k}')
        return files[k]
    return f


class KeepLast(unittest.TestCase):
    def setUp(self):
        self.real = kl.fetch

    def tearDown(self):
        kl.fetch = self.real

    def test_puts_back_the_whole_water_temperature(self):
        kl.fetch = fake(PUBLISHED)
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, 'watertemp'))
            with open(os.path.join(d, 'watertemp', 't9_9.png'), 'wb') as fh:   # a step that stopped partway
                fh.write(b'half')
            kl.keep_watertemp(d, SITE, say=lambda *x: None)
            got = sorted(os.listdir(os.path.join(d, 'watertemp')))
            self.assertEqual(got, ['manifest.json', 'o.png', 'stations.json', 't5_3.png', 't6_3.png'])
            with open(os.path.join(d, 'watertemp', 't6_3.png'), 'rb') as fh:
                self.assertEqual(fh.read(), b'tile63')

    def test_no_manifest_without_every_picture_it_lists(self):
        files = dict(PUBLISHED)
        del files['watertemp/t6_3.png']
        kl.fetch = fake(files)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OSError):
                kl.keep_watertemp(d, SITE, say=lambda *x: None)
            self.assertFalse(os.path.exists(os.path.join(d, 'watertemp', 'manifest.json')))

    def test_keeps_the_pictures_without_the_stations(self):
        files = dict(PUBLISHED)
        del files['watertemp/stations.json']
        kl.fetch = fake(files)
        with tempfile.TemporaryDirectory() as d:
            kl.keep_watertemp(d, SITE, say=lambda *x: None)
            self.assertTrue(os.path.exists(os.path.join(d, 'watertemp', 'manifest.json')))
            self.assertFalse(os.path.exists(os.path.join(d, 'watertemp', 'stations.json')))

    def test_puts_back_the_inlets(self):
        kl.fetch = fake(PUBLISHED)
        with tempfile.TemporaryDirectory() as d:
            kl.keep_inlets(d, SITE, say=lambda *x: None)
            with open(os.path.join(d, 'inlets.json'), 'rb') as fh:
                self.assertEqual(fh.read(), PUBLISHED['inlets.json'])


if __name__ == '__main__':
    unittest.main()
