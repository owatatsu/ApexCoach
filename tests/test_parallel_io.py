import numpy as np

from apexcoach.pipeline import AsyncFrameWriter, PrefetchFrameStream


class DummyWriter:
    def __init__(self) -> None:
        self.frames = []
        self.released = False

    def write(self, frame) -> None:
        self.frames.append(frame)

    def release(self) -> None:
        self.released = True


def test_prefetch_stream_yields_all_items() -> None:
    stream = PrefetchFrameStream(iter([1, 2, 3]), queue_size=2)
    got = list(stream)
    stream.close()
    assert got == [1, 2, 3]


def test_async_writer_writes_and_releases() -> None:
    writer = DummyWriter()
    async_writer = AsyncFrameWriter(writer=writer, queue_size=2)
    async_writer.write(np.zeros((2, 2, 3), dtype=np.uint8))
    async_writer.write(np.ones((2, 2, 3), dtype=np.uint8))
    async_writer.close()
    assert len(writer.frames) == 2
    assert writer.released is True
