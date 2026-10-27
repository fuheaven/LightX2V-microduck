"""Starting point: retain the official robot and independent task evaluator."""

from lightx2v_microduck.studio.plugins import FindObject


class MyFindObject(FindObject):
    name = "my_find_object"
    version = "0.1.0"
    # Override teacher() to supply a demonstrator, or environment() for new assets.
    # Change score() only with an explicitly versioned task definition.
