import unittest
import encode_decode


class EncoderDecodeTest(unittest.TestCase):
    def testJP(self):
        jp = ["abc"]  # doesn't actually need to be japanese, it's all character based
        enc = encode_decode.JPEncoderDecoder(jp)
        actual = enc.encode("bac")
        self.assertEqual(len(actual), encode_decode.DEFAULT_BLOCK_SIZE)
        self.assertListEqual([encode_decode.BOS_I, 4, 3, 5, 1], actual[:5])

    def testJPRoundTrip(self):
        jps = ["abc", "def", "ghi"]
        enc = encode_decode.JPEncoderDecoder(jps)
        for jp in jps:
            actual = enc.decode(enc.encode(jp))
            self.assertIn(jp, actual)
            self.assertEqual(actual[:3], encode_decode.BOS)
            self.assertEqual(
                actual[
                    len(encode_decode.BOS)
                    + len(jp) : len(encode_decode.BOS)
                    + len(jp)
                    + len(encode_decode.EOS)
                ],
                encode_decode.EOS,
            )

    def testEn(self):
        en = ["abab"]
        enc = encode_decode.ENEncoderDecoder(en, 1)
        actual = enc.encode("ab")
        self.assertEqual(len(actual), encode_decode.DEFAULT_BLOCK_SIZE)
        self.assertListEqual([encode_decode.BOS_I, 5, 1], actual[:3])

    def testENRoundTrip(self):
        ens = ["abc", "def", "ghi"]
        enc = encode_decode.ENEncoderDecoder(ens, 5)
        for en in ens:
            actual = enc.decode(enc.encode(en))
            self.assertIn(en, actual)
            self.assertEqual(actual[:3], encode_decode.BOS)
            self.assertEqual(
                actual[
                    len(encode_decode.BOS)
                    + len(en) : len(encode_decode.BOS)
                    + len(en)
                    + len(encode_decode.EOS)
                ],
                encode_decode.EOS,
            )
    def testENCorrectlyCountsRepeatedMerges(self):
        # ab: 1
        # bb: 2
        # bc: 1
        ens = ["abbbc"]
        enc = encode_decode.ENEncoderDecoder(ens, 5)
        # Tests internal members
        base = len(encode_decode.TOKEN_MAPPING)
        self.assertDictContainsSubset({**encode_decode.TOKEN_MAPPING, **{'a': base, 'b': base+1, 'c': base+2, 'bb': base+3}}, enc.merges)

if __name__ == "__main__":
    unittest.main()
