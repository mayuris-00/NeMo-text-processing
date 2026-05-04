# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.hi.graph_utils import (
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    GraphFst,
    delete_space,
)


class ElectronicFst(GraphFst):
    """
    Finite state transducer for verbalizing electronic tokens → written form.

    Protocol key → written separator (exact inverse of protocols.tsv):
        "https"    → "https://"      "http"     → "http://"
        "www"      → "www."          "httpswww" → "https://www."
        "httpwww"  → "http://www."

    Args:
        deterministic: if True will provide a single transduction option
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="verbalize", deterministic=deterministic)

        field_val = pynini.closure(NEMO_NOT_QUOTE, 1)

        del_username = pynutil.delete("username: \"")
        del_domain   = pynutil.delete("domain: \"")
        del_protocol = pynutil.delete("protocol: \"")
        del_path     = pynutil.delete("path: \"")
        del_quote    = pynutil.delete("\"")
        del_sp       = delete_space

        # Longer keys first so "httpswww" matches before "https"
        protocol_reconstruct = (
            pynutil.add_weight(pynini.cross("httpswww", "https://www."), 0.9)
            | pynutil.add_weight(pynini.cross("httpwww",  "http://www."),  0.9)
            | pynutil.add_weight(pynini.cross("https",    "https://"),      1.0)
            | pynutil.add_weight(pynini.cross("http",     "http://"),       1.0)
            | pynutil.add_weight(pynini.cross("www",      "www."),          1.0)
        )

        username_graph = del_username + field_val + del_quote
        domain_graph   = del_domain   + field_val + del_quote
        protocol_graph = del_protocol + protocol_reconstruct + del_quote
        path_graph_fld = del_path     + field_val + del_quote

        # URL:    "https://" + "google.com"  →  "https://google.com"
        url_graph = protocol_graph + del_sp + domain_graph

        # EMAIL:  "kumar" + "@" + "gmail.com"  →  "kumar@gmail.com"
        email_graph = username_graph + del_sp + pynutil.insert("@") + domain_graph

        path_only   = path_graph_fld
        domain_only = domain_graph

        graph = (
            pynutil.add_weight(url_graph,    1.0)
            | pynutil.add_weight(email_graph,  1.01)
            | pynutil.add_weight(path_only,    1.02)
            | pynutil.add_weight(domain_only,  1.03)
        )

        self.fst = self.delete_tokens(graph).optimize()
