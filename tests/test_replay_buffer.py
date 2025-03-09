import unittest


class MyTestCase(unittest.TestCase):
    def test_replay_buffer_dataloader(self):
        import numpy as np
        from utils.data_util import EfficientReplayBuffer
        n, s, a, c, h, w = np.random.randint(1024, 2048), 6, 2, 3, 64, 64
        data = dict(
            images=np.random.randint(0, 256, size=(n, c, h, w), dtype=np.uint8),
            states=np.random.randn(n, s).astype(np.float32),
            actions=np.random.randn(n, a).astype(np.float32),
            rewards=np.random.randn(n, ).astype(np.float32),
            dones=np.random.uniform(0, 1, (n,)) > 0.9,
            status=np.random.randint(-1, 4, (n, )).astype(np.int64)
        )

        testcases = [
            (
                ['states'], ['actions']
            ),
            (
                ['images', 'states'], ['actions', 'safe']
            ),
            (
                ['images', 'states'], ['actions', 'rewards', 'safe'], ['dones', 'status']
            )
        ]

        constants = {
            'safe': True
        }

        for manifest in testcases:
            with self.subTest(manifest=manifest):
                buffer = EfficientReplayBuffer(constants=constants)
                for i in range(n):
                    buffer.append(batched=False, **{k: v[i] for k, v in data.items()})

                batch_size = np.random.randint(32, 64)
                dataloader = buffer.dataloader(batch_size=batch_size, shuffle=False, manifest=manifest)
                for minibatch, all_features in enumerate(dataloader):
                    self.assertEqual(len(all_features), len(manifest))
                    for features, feature_names in zip(all_features, manifest):
                        self.assertEqual(len(features), len(feature_names))
                        for feature, feature_name in zip(features, feature_names):
                            if feature_name in data:
                                self.assertTrue(np.allclose(feature.cpu().numpy(), data[feature_name][minibatch * batch_size:(minibatch + 1) * batch_size]))
                                self.assertTrue(feature.cpu().numpy().dtype == data[feature_name].dtype)
                            elif feature_name in constants:
                                self.assertTrue(np.allclose(feature.cpu().numpy(), constants[feature_name]))

    def test_replay_buffer_knn_cvx(self):
        import numpy as np
        from utils.data_util import EfficientReplayBuffer
        from scipy.spatial import ConvexHull
        n, s, a, c, h, w = np.random.randint(1024, 2048), 6, 2, 3, 64, 64
        q = 100
        k = 10
        data = dict(
            images=np.random.randint(0, 256, size=(n, c, h, w), dtype=np.uint8),
            states=np.random.randn(n, s).astype(np.float32),
            actions=np.random.randn(n, a).astype(np.float32),
            rewards=np.random.randn(n, ).astype(np.float32),
            dones=np.random.uniform(0, 1, (n,)) > 0.9,
            status=np.random.randint(-1, 4, (n, )).astype(np.int64)
        )
        buffer = EfficientReplayBuffer(maxsize=n * 2, constants={'safe': True})
        buffer.append(batched=True, size=n, **data)

        testcases = (
            ('states',),
            ('states', 'actions'),
        )

        for fields in testcases:
            with self.subTest(fields=fields, type='in'):
                size = sum(data[field][0].size for field in fields)
                selected_features = np.concatenate([data[field] for field in fields], axis=-1)
                hull = ConvexHull(selected_features)
                features = selected_features[hull.vertices]

                # Generate points in the convex hull.
                weights = np.random.rand(q, features.shape[0], 1)
                weights /= np.sum(weights, axis=1, keepdims=True)
                queries = np.mean((weights * features), axis=1)
                self.assertTrue(queries.shape == (q, size), f"{queries.shape}")

                ret = buffer.is_in_knn_convex_hull(queries, fields, 32, threshold=np.inf)
                self.assertTrue(ret.all())

            with self.subTest(fields=fields, type='out'):
                # Generate points out of the convex hull.
                queries = []
                for _ in range(q):
                    centeroid = np.mean(features, axis=0)
                    random_vertex = selected_features[np.random.choice(hull.vertices)]
                    queries.append(centeroid * -0.5 + random_vertex * 1.5)
                queries = np.asarray(queries)
                self.assertTrue(queries.shape == (q, size), f"{queries.shape}")

                ret = buffer.is_in_knn_convex_hull(queries, fields, 32, threshold=np.inf)
                self.assertFalse(ret.any())


if __name__ == '__main__':
    unittest.main()
