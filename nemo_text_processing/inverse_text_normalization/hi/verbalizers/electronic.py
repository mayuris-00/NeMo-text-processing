# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2024 and onwards Google, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.hi.graph_utils import (
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_space,
)


class ElectronicFst(GraphFst):
    """
    Finite state transducer for verbalizing electronic addresses (ITN direction).
    Reads the structured tagged token and reconstructs the original written form.

    Examples:
        electronic { username: "kumar" domain: "gmail.com" }  ->  kumar@gmail.com
        electronic { protocol: "https" domain: "google.com" } ->  https://google.com
        electronic { domain: "google.com" }                   ->  google.com
        electronic { domain: "192.168.1.1" }                  ->  192.168.1.1

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transductions are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="verbalize", deterministic=deterministic)

        # ==================== FIELD DELETERS ====================
        delete_username_tag = pynutil.delete("username: \"")
        delete_domain_tag = pynutil.delete("domain: \"")
        delete_protocol_tag = pynutil.delete("protocol: \"")
        delete_quote = pynutil.delete("\"")

        # Content inside quotes: any non-quote character
        content = pynini.closure(NEMO_NOT_QUOTE, 1)

        # ==================== FIELD GRAPHS ====================

        # username: "kumar"  ->  kumar
        username_graph = delete_username_tag + content + delete_quote

        # domain: "gmail.com"  ->  gmail.com
        domain_graph = delete_domain_tag + content + delete_quote

        # protocol: "https"  ->  https://
        # The protocol value stored is just "https" / "http" / "www"
        # We need to reconstruct the full prefix: https:// or http:// or www.
        protocol_value = pynini.union(
            pynini.cross("https", "https://"),
            pynini.cross("http", "http://"),
            pynini.cross("httpswww", "https://www."),
            pynini.cross("httpwww", "http://www."),
            pynini.cross("www", "www."),
        )
        protocol_graph = delete_protocol_tag + protocol_value + delete_quote

        # ==================== COMBINED GRAPH ====================

        # Email: username: "kumar" domain: "gmail.com"  ->  kumar@gmail.com
        email_graph = username_graph + pynutil.insert("@") + delete_space + domain_graph

        # URL: protocol: "https" domain: "google.com"  ->  https://google.com
        url_graph = protocol_graph + delete_space + domain_graph

        # Domain only: domain: "google.com"  ->  google.com
        domain_only_graph = domain_graph

        graph = (
            pynutil.add_weight(url_graph, 1.0)
            | pynutil.add_weight(email_graph, 1.01)
            | pynutil.add_weight(domain_only_graph, 1.02)
        )

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
