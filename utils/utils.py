"""Utility functions"""
import struct
from colorama import Fore
from nitdms.common import LeadIn, KToC
from nitdms.exceptions import SegmentCorruptedError


def print_hex(tdms_file):
    """Print the raw bytes of the TDMS file in hexadecimal

    Bytes colored according to type:
        Leadin - red
        Meta data - green
        Raw data - cyan

    Args:
        tdms_file (str): TDMS filename
    """

    def get_leadins(buffer, ptr):
        """Leadin generator function"""
        while ptr < len(buffer):
            toc = struct.unpack("<I", buffer[ptr + 4 : ptr + 8])[0]
            byte_order = ">" if toc & KToC.BigEndian else "<"
            fmt = byte_order + "QQ"
            ptr += 12
            leadin = LeadIn(toc, *struct.unpack(fmt, buffer[ptr : ptr + 16]))
            ptr += 16
            if ptr + leadin.segment_len > len(buffer):
                raise SegmentCorruptedError
            metadata_start = ptr
            yield leadin
            if ptr != metadata_start + leadin.metadata_len:
                ptr += metadata_start + leadin.metadata_len - ptr
            # move the pointer to beginning of next segment
            ptr += leadin.segment_len - leadin.metadata_len

    def format_hex(buffer, color):
        """Convert bytes to colorized hex string"""
        return color + "".join(f"{b:02x} " for b in buffer) + Fore.RESET

    with open(tdms_file, "rb") as f:
        buffer = f.read()
    ptr = 0
    partial_line = ""
    remaining = 0
    lines = []
    for leadin in get_leadins(buffer, ptr):
        if remaining > 0:
            partial_line += format_hex(buffer[ptr : ptr + remaining], Fore.RED)
            lines.append(partial_line)
        end = ptr + 28
        ptr += remaining
        n_lines, remainder = divmod(end - ptr, 16)
        remaining = 16 - remainder
        for _ in range(n_lines):
            lines.append(format_hex(buffer[ptr : ptr + 16], Fore.RED))
            ptr += 16
        partial_line = format_hex(buffer[ptr:end], Fore.RED)
        ptr = end
        if leadin.metadata_len > 0:
            end = ptr + leadin.metadata_len
            partial_line += format_hex(buffer[ptr : ptr + remaining], Fore.GREEN)
            lines.append(partial_line)
            ptr += remaining
            n_lines, remainder = divmod(end - ptr, 16)
            remaining = 16 - remainder
            for _ in range(n_lines):
                lines.append(format_hex(buffer[ptr : ptr + 16], Fore.GREEN))
                ptr += 16
            partial_line = format_hex(buffer[ptr:end], Fore.GREEN)
            ptr = end
        if leadin.segment_len - leadin.metadata_len > 0:
            partial_line += format_hex(buffer[ptr : ptr + remaining], Fore.CYAN)
            lines.append(partial_line)
            end = ptr + leadin.segment_len - leadin.metadata_len
            ptr += remaining
            n_lines, remainder = divmod(end - ptr, 16)
            remaining = 16 - remainder
            for _ in range(n_lines):
                lines.append(format_hex(buffer[ptr : ptr + 16], Fore.CYAN))
                ptr += 16
            partial_line = format_hex(buffer[ptr:end], Fore.CYAN)
            ptr = end
    lines.append(partial_line)
    width = len(str(16 * len(lines)))
    for n, line in enumerate(lines):
        print(f"{n*16:<{width}d}: {line}")
