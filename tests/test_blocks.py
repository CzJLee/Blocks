import unittest
import parameterized
from blocks.blocks import Piece

NON_CONNECTED_DIAGONAL_PIECE = Piece({(0, 0, 0), (1, 1, 1)}, validate=False)
NON_CONNECTED_EDGE_PIECE = Piece({(0, 0, 0), (1, 1, 0)}, validate=False)
CORNER_PIECE = Piece({(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)})
SQUARE_PIECE = Piece({(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)})
CUBIT_PIECE = Piece({(0, 0, 0)})


class TestPiece(unittest.TestCase):

    @parameterized.parameterized.expand(
        [
            parameterized.param("corner_piece", piece=CORNER_PIECE, expected=4),
            parameterized.param("square_piece", piece=SQUARE_PIECE, expected=4),
            parameterized.param("cubit_piece", piece=CUBIT_PIECE, expected=1),
        ],
    )
    def test_len(self, _, piece: Piece, expected: int):
        self.assertEqual(len(piece), expected)

    def test_translate(self):
        p = Piece({(0, 0, 0), (1, 0, 0)})
        translated = p.translate(1, 1, 1)
        self.assertEqual(translated, Piece({(1, 1, 1), (2, 1, 1)}, canonicalize=False))

    def test_rotate_x(self):
        """Test rotating a piece around the x-axis"""
        rotated = SQUARE_PIECE.rotate_x(canonicalize=False)
        self.assertEqual(rotated, Piece([(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1)]))

        rotated_c = SQUARE_PIECE.rotate_x(canonicalize=True)
        self.assertEqual(rotated_c, Piece({(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1)}))

    def test_rotate_y(self):
        """Test rotating a piece around the y-axis"""
        rotated = SQUARE_PIECE.rotate_y(canonicalize=False)
        self.assertEqual(rotated, Piece([(0, 0, -1), (0, 0, 0), (0, 1, -1), (0, 1, 0)]))

        rotated_c = SQUARE_PIECE.rotate_y(canonicalize=True)
        self.assertEqual(rotated_c, Piece([(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]))

    def test_rotate_z(self):
        """Test rotating a piece around the z-axis"""
        rotated = SQUARE_PIECE.rotate_z(canonicalize=False)
        self.assertEqual(rotated, Piece([(0, 0, 0), (0, 1, 0), (-1, 0, 0), (-1, 1, 0)]))

        rotated_c = SQUARE_PIECE.rotate_z(canonicalize=True)
        self.assertEqual(rotated_c, Piece({(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)}))

    @parameterized.parameterized.expand(
        [
            parameterized.param(
                "non_connected_diagonal_piece",
                piece=NON_CONNECTED_DIAGONAL_PIECE,
                expected=False,
            ),
            parameterized.param(
                "non_connected_edge_piece",
                piece=NON_CONNECTED_EDGE_PIECE,
                expected=False,
            ),
            parameterized.param("corner_piece", piece=CORNER_PIECE, expected=True),
            parameterized.param("square_piece", piece=SQUARE_PIECE, expected=True),
        ],
    )
    def test_is_contiguous(self, _, piece: Piece, expected: bool):
        self.assertEqual(piece.is_contiguous(), expected)

    @parameterized.parameterized.expand(
        [
            parameterized.param(
                "congruent_under_translation",
                piece_a=CORNER_PIECE,
                piece_b=CORNER_PIECE.translate(1, 1, 1),
                expected=True,
            ),
            parameterized.param(
                "congruent_under_rotation",
                piece_a=CORNER_PIECE,
                piece_b=CORNER_PIECE.rotate_x(),
                expected=True,
            ),
            parameterized.param(
                "congruent_under_translation_and_rotation",
                piece_a=CORNER_PIECE,
                piece_b=CORNER_PIECE.translate(1, 2, 3).rotate_x().rotate_y(),
                expected=True,
            ),
            parameterized.param(
                "not_congruent_pieces",
                piece_a=CORNER_PIECE,
                piece_b=SQUARE_PIECE,
                expected=False,
            ),
        ],
    )
    def test_is_congruent(self, _, piece_a: Piece, piece_b: Piece, expected: bool):
        self.assertEqual(piece_a.is_congruent(piece_b), expected)


if __name__ == "__main__":
    unittest.main()
