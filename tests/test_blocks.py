import unittest
import parameterized
from blocks.blocks import Piece

NON_CONNECTED_DIAGONAL_PIECE = Piece({(0, 0, 0), (1, 1, 1)}, validate=False)
NON_CONNECTED_EDGE_PIECE = Piece({(0, 0, 0), (1, 1, 0)}, validate=False)
CORNER_PIECE = Piece({(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)})
SQUARE_PIECE = Piece({(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)})


class TestPiece(unittest.TestCase):

    def test_translate(self):
        p = Piece({(0, 0, 0), (1, 0, 0)})
        translated = p.translate(1, 1, 1)
        self.assertEqual(translated, Piece({(1, 1, 1), (2, 1, 1)}, canonicalize=False))

    def test_rotate_x(self):
        """Test rotating a piece around the x-axis"""
        rotated = SQUARE_PIECE.rotate_x(canonicalize=False)
        self.assertEqual(rotated, Piece({(0, 0, 0), (1, 0, 0), (0, 0, -1), (1, 0, -1)}))

        rotated_c = SQUARE_PIECE.rotate_x(canonicalize=True)
        self.assertEqual(rotated_c, Piece({(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1)}))

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
    def test_is_contiguous(
        self,
        _,
        piece: Piece,
        expected: bool,
    ):
        self.assertEqual(piece.is_contiguous(), expected)


if __name__ == "__main__":
    unittest.main()
