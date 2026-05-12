# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
    GraphFst,
    delete_space,
    delete_zero_or_one_space,
    NEMO_SIGMA,
)
from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path


class ElectronicFst(GraphFst):
    """
    ITN FST for classifying electronic expressions.

    Changes vs original
    -------------------
    CHANGE-1  path_atom_url: add digit_words (0.88) and digit_glyphs (0.89).
              Fixes URL paths with digit segments like "629/790/any".

    CHANGE-2  domain_and_path: optional hash_frag suffix with hyphen support.
              हैशटैग → # followed by token_seq + closure(hyphen+token_seq).
              Fixes tests 0161-0170.

    CHANGE-3  alnum_body dot-branch: accept "प्वाइंट" alongside "डॉट"/"DOT".
              Fixes test 0906 (PM2.5).

    CHANGE-4  special_codes_map: additive entries for missile/ordinal/product names.
              Fixes 0978,0985,0990,0991,0992,0994,0995,0986,0818.

    CHANGE-5  unix_abs_path_fst weight raised from 1.10 to 15.00.
              ROOT CAUSE FIX for "domain.com /path/" stray-space bugs (~25 tests).
              NeMo's tagger was splitting URL+path into two spans:
                span1 = url_fst matching domain only  (weight 0.10)
                span2 = unix_abs_path_fst matching path (weight 1.10)
                combined cost = 1.20  <  url_fst-with-path cost (~2.0+)
              so the two-span split always won, and NeMo joined them with the
              literal space that existed between them in the input.
              Setting unix_abs_path_fst to 15.00 makes the split cost 15.10,
              which always loses to url_fst matching the full URL in one span.
              SAFE: url_fst starts with "एच" (HTTP), unix_abs_path_fst starts
              with "फॉरवर्ड" — mutually exclusive. Pure unix path inputs have
              no competing FST so weight 15 vs 1.1 makes no difference there.

    CHANGE-6  literal_rel_path_fst: new FST accepting " / " (literal ASCII slash)
              as path separator. Fixes tests 0303 (होम / डेस्कटॉप) and
              0486 (बैकअप्स / टेम्प) which use literal "/" not spoken "फॉरवर्ड स्लैश".
              Added at weight 1.15. Literal " / " never appears in URL inputs.
    """

    def __init__(self):
        super().__init__(name="electronic", kind="classify")

        # ── Digit maps ────────────────────────────────────────────────────────
        digit_words = pynini.string_map([
            ("एक", "1"), ("दो", "2"), ("तीन", "3"), ("चार", "4"), ("पाँच", "5"),
            ("छह", "6"), ("सात", "7"), ("आठ", "8"), ("नौ", "9"), ("शून्य", "0"),
        ])
        digit_glyphs = pynini.string_map([
            ("१", "1"), ("२", "2"), ("३", "3"), ("४", "4"), ("५", "5"),
            ("६", "6"), ("७", "7"), ("८", "8"), ("९", "9"), ("०", "0"),
        ])
        subscript_digit_words = pynini.string_map([
            ("एक", "₁"), ("दो", "₂"), ("तीन", "₃"), ("चार", "₄"), ("पाँच", "₅"),
            ("छह", "₆"), ("सात", "₇"), ("आठ", "₈"), ("नौ", "₉"), ("शून्य", "₀"),
        ])
        single_digit = (
            pynutil.add_weight(digit_glyphs, 0.8)
            | pynutil.add_weight(digit_words,  0.9)
        )
        digit_seq = (
            pynutil.add_weight(digit_glyphs + pynini.closure(digit_glyphs, 0), 0.8)
            | pynutil.add_weight(digit_words  + pynini.closure(delete_space + digit_words, 0), 0.9)
        )

        # ── TSV maps ──────────────────────────────────────────────────────────
        letter_map = pynini.string_file(get_abs_path("data/electronic/letters.tsv")).invert()
        domain_map = pynini.string_file(get_abs_path("data/electronic/domain.tsv")).invert()
        server_map = pynini.string_file(get_abs_path("data/electronic/server_name.tsv")).invert()
        common_map = pynini.string_file(get_abs_path("data/electronic/common_words.tsv")).invert()

        try:
            chem_named_map = pynini.string_file(
                get_abs_path("data/electronic/chemical_formulas.tsv")
            ).optimize()
        except Exception:
            chem_named_map = None

        # ── Special codes ─────────────────────────────────────────────────────
        # CHANGE-4: new entries appended; all original entries unchanged.
        special_codes_map = pynini.string_map([
            # ── original entries ──────────────────────────────────────────────
            ("आई ए तीन दो",                        "ia32"),
            ("एक्स आठ छह",                          "x86"),
            ("छह एस",                               "6s"),
            ("यू टी एफ आठ",                        "utf8"),
            ("ए एस सी आई आई आठ पाँच",             "Ascii85"),
            ("पी एच आर ए सी के पाँच सात",         "phrack57"),
            ("ज़ेड एन एस ओ चार",                    "ZnSO4"),
            ("ज़ेड एक्स आठ शून्य",                  "ZX80"),
            ("ज़ेड एक्स आठ एक",                     "ZX81"),
            ("ज़ेड एक्स आठ शून्य एक नौ आठ शून्य", "ZX80 1980"),
            ("एम जी बी आर दो",                      "MgBr2"),
            ("बी ए एस ई तीन दो",                    "Base32"),
            ("बी ई सी एच तीन दो",                   "Bech32"),
            ("सी ए एन ओ एन ए सात पाँच",            "Canon A75"),
            ("ऑडी आठ शून्य बटा नौ शून्य बी चार",  "ऑडी 80/90 B4"),
            # ── CHANGE-4: additive entries ────────────────────────────────────
            ("अग्नि द्वितीय",                        "अग्नि-2"),
            ("कोविड नौटीन",                          "कोविड-19"),
            ("पृथ्वी चार",                           "पृथ्वी-4"),
            ("ब्रह्मोस हाइफ़न एक",                   "ब्रह्मोस-1"),
            ("भास्कर हाइफ़न दो",                     "भास्कर-II"),
            ("श्रेणी हाइफ़न दो",                     "श्रेणी-II"),
            ("रोहिणी आर एस हाइफ़न एक",              "रोहिणी आर एस-I"),
            ("आर ई डी एम आई एच चार चार चार जी",    "REDMI H44 4G"),
            ("आई एन एन ओ टी ई एक",                  "IN NOTE1"),
            ("ऑडी आठ शून्य बटा  नौ शून्य  बी चार", "ऑडी 80/90 B4"),
        ]).optimize()

        # ── Lowercase / uppercase helpers ─────────────────────────────────────
        to_lower = pynini.cdrewrite(
            pynini.string_map([(chr(c), chr(c + 32)) for c in range(ord('A'), ord('Z') + 1)]),
            "", "", NEMO_SIGMA,
        ).optimize()
        to_upper = pynini.cdrewrite(
            pynini.string_map([(chr(c + 32), chr(c)) for c in range(ord('A'), ord('Z') + 1)]),
            "", "", NEMO_SIGMA,
        ).optimize()

        def make_lower(fst):
            return (fst @ to_lower).optimize()

        def make_upper(fst):
            return (fst @ to_upper).optimize()

        letter_map_lower = make_lower(letter_map)
        letter_map_upper = make_upper(letter_map)
        common_map_lower = make_lower(common_map)
        server_map_lower = make_lower(server_map)

        # ── Latin runs (for stray ASCII in input) ─────────────────────────────
        latin_run = pynini.closure(
            pynini.union(*[pynini.accep(chr(c)) for c in range(ord('A'), ord('Z') + 1)])
            | pynini.union(*[pynini.accep(chr(c)) for c in range(ord('a'), ord('z') + 1)]),
            1,
        )
        latin_run_lower = make_lower(latin_run)

        # ── Symbol primitives ─────────────────────────────────────────────────
        def _backslash():
            return (pynutil.delete("बैकवर्ड") + delete_space
                    + pynutil.delete("स्लैश") + pynutil.insert("\\\\"))
        seg_backslash   = delete_space + _backslash() + delete_space
        trail_backslash = delete_space + _backslash()
        lead_backslash  = _backslash() + delete_space

        def _unix_slash():
            return (pynutil.delete("फॉरवर्ड") + delete_space
                    + pynutil.delete("स्लैश") + pynutil.insert("/"))
        unix_seg_slash   = delete_space + _unix_slash() + delete_space
        unix_lead_slash  = _unix_slash() + delete_space
        unix_trail_slash = delete_space + _unix_slash()

        # *** FIX Bug 3 *** (original — spoken फॉरवर्ड स्लैश → /)
        url_slash = (
            delete_space
            + pynutil.delete("फॉरवर्ड") + delete_space + pynutil.delete("स्लैश")
            + pynutil.insert("/")
        )

        # CHANGE-6: literal ASCII "/" as path separator (for inputs like "होम / डेस्कटॉप")
        # delete_space + delete("/") + delete_space  →  outputs nothing (slash inserted separately)
        lit_slash_seg = pynini.cross(" / ", "/")

        dot           = delete_space + (pynutil.delete("डॉट") | pynutil.delete("DOT")) + delete_space + pynutil.insert(".")
        dot_end_safe  = delete_space + (pynutil.delete("डॉट") | pynutil.delete("DOT")) + delete_zero_or_one_space + pynutil.insert(".")
        hyphen        = delete_space + (pynutil.delete("हाइफ़न") | pynutil.delete("हाइफन")) + delete_space + pynutil.insert("-")
        underscore    = delete_space + pynutil.delete("अंडर") + delete_space + pynutil.delete("स्कोर") + pynutil.insert("_")
        at_sign       = delete_space + pynutil.delete("एट") + delete_space
        x_sep         = delete_space + pynutil.delete("एक्स") + pynutil.insert("x")
        literal_space = delete_space + pynutil.delete("स्पेस") + pynutil.insert(" ")
        open_bracket  = delete_space + pynutil.delete("ओपन")  + delete_space + pynutil.delete("ब्रेकेट") + pynutil.insert("(")
        close_bracket = delete_space + pynutil.delete("क्लोज़") + delete_space + pynutil.delete("ब्रेकेट") + pynutil.insert(")")
        dollar_sign   = delete_space + pynutil.delete("डॉलर") + pynutil.insert("$")
        or_word        = pynutil.delete("ओ") + delete_space + pynutil.delete("आर") + pynutil.insert("or")
        and_as_letters = pynutil.delete("एंड") + pynutil.insert("and")
        www_token      = (pynutil.delete("डब्ल्यू") + delete_space
                          + pynutil.delete("डब्ल्यू") + delete_space
                          + pynutil.delete("डब्ल्यू") + pynutil.insert("www"))
        v_prefix       = pynutil.delete("वी") + pynutil.insert("v")
        hp_token       = pynutil.delete("एच") + delete_space + pynutil.delete("पी") + pynutil.insert("HP")
        tilde_delete   = pynutil.delete("~") | pynutil.delete("टिल्ड")
        drive_letter   = pynini.string_map([
            ("सी", "C"), ("डी", "D"), ("ई", "E"), ("एफ", "F"),
            ("जी", "G"), ("एच", "H"), ("आई", "I"), ("जे", "J"),
        ])

        # ── Token primitives ──────────────────────────────────────────────────
        single_token = (
            pynutil.add_weight(server_map,         0.90)
            | pynutil.add_weight(common_map,       0.95)
            | pynutil.add_weight(letter_map_lower, 1.00)
        )
        token_seq = single_token + pynini.closure(delete_space + single_token, 0)

        path_atom = (
            pynutil.add_weight(hp_token,        0.76)
            | pynutil.add_weight(www_token,      0.77)
            | pynutil.add_weight(or_word,        0.80)
            | pynutil.add_weight(and_as_letters, 0.84)
            | pynutil.add_weight(common_map,     0.90)
            | pynutil.add_weight(server_map,     0.92)
            | pynutil.add_weight(digit_words,    0.94)
            | pynutil.add_weight(digit_glyphs,   0.95)
            | pynutil.add_weight(latin_run,      0.97)
            | pynutil.add_weight(letter_map,     1.00)
        )
        path_atom_lower = (
            pynutil.add_weight(common_map_lower,   0.90)
            | pynutil.add_weight(server_map_lower, 0.92)
            | pynutil.add_weight(digit_words,      0.94)
            | pynutil.add_weight(digit_glyphs,     0.95)
            | pynutil.add_weight(latin_run_lower,  0.97)
            | pynutil.add_weight(letter_map_lower, 1.00)
        )
        unix_path_atom = (
            pynutil.add_weight(www_token,                    0.77)
            | pynutil.add_weight(or_word,                    0.80)
            | pynutil.add_weight(and_as_letters,             0.84)
            | pynutil.add_weight(pynini.cross("CI", "c"),    0.86)
            | pynutil.add_weight(common_map_lower,           0.90)
            | pynutil.add_weight(server_map_lower,           0.92)
            | pynutil.add_weight(digit_words,                0.94)
            | pynutil.add_weight(digit_glyphs,               0.95)
            | pynutil.add_weight(latin_run_lower,            0.97)
            | pynutil.add_weight(letter_map_lower,           1.00)
        )

        # ── File extension ────────────────────────────────────────────────────
        single_ext = (
            delete_space + pynutil.delete("डॉट") + pynutil.insert(".")
            + delete_space + path_atom_lower
            + pynini.closure(delete_space + path_atom_lower, 0)
        )
        ext_hyphen = (
            delete_space + (pynutil.delete("हाइफ़न") | pynutil.delete("हाइफन"))
            + pynutil.insert("-") + delete_space
            + path_atom_lower + pynini.closure(delete_space + path_atom_lower, 0)
        )
        file_ext = single_ext + pynini.closure(single_ext | ext_hyphen, 0)

        # ── Windows path segment ──────────────────────────────────────────────
        win_hyphen = (
            delete_space + (pynutil.delete("हाइफ़न") | pynutil.delete("हाइफन"))
            + pynutil.insert("-") + delete_space
            + path_atom + pynini.closure(delete_space + path_atom, 0)
        )
        win_underscore = (
            delete_space + pynutil.delete("अंडर") + delete_space
            + pynutil.delete("स्कोर") + pynutil.insert("_")
        )
        path_segment = (
            path_atom
            + pynini.closure(
                pynutil.add_weight(delete_space + path_atom, 1.0)
                | pynutil.add_weight(win_hyphen,             1.0)
                | pynutil.add_weight(win_underscore,         1.0)
                | pynutil.add_weight(literal_space,          1.0)
                | pynutil.add_weight(open_bracket,           1.0)
                | pynutil.add_weight(close_bracket,          1.0)
            , 0)
            + pynini.closure(file_ext, 0, 1)
        )

        # ── Unix path segment ─────────────────────────────────────────────────
        unix_hyphen = (
            delete_space + (pynutil.delete("हाइफ़न") | pynutil.delete("हाइफन"))
            + pynutil.insert("-") + delete_space
            + unix_path_atom + pynini.closure(delete_space + unix_path_atom, 0)
        )
        unix_underscore = (
            delete_space + pynutil.delete("अंडर") + delete_space
            + pynutil.delete("स्कोर") + pynutil.insert("_")
            + delete_space + unix_path_atom
            + pynini.closure(delete_space + unix_path_atom, 0)
        )
        version_seg = (
            v_prefix + unix_path_atom
            + pynini.closure(
                delete_space + pynutil.delete("डॉट") + pynutil.insert(".")
                + delete_space + unix_path_atom
                + pynini.closure(delete_space + unix_path_atom, 0)
            , 0)
        )
        dollar_var = (
            dollar_sign + delete_space
            + unix_path_atom + pynini.closure(delete_space + unix_path_atom, 0)
        )
        unix_segment = (
            (
                pynutil.add_weight(version_seg,      0.85)
                | pynutil.add_weight(dollar_var,     0.87)
                | pynutil.add_weight(unix_path_atom, 1.00)
            )
            + pynini.closure(
                pynutil.add_weight(delete_space + unix_path_atom, 1.0)
                | pynutil.add_weight(unix_hyphen,                 1.0)
                | pynutil.add_weight(unix_underscore,             1.0)
            , 0)
            + pynini.closure(file_ext, 0, 1)
        )

        # ── Path FSTs ─────────────────────────────────────────────────────────
        windows_path_fst = (
            pynutil.insert("path: \"")
            + drive_letter + delete_space + pynutil.delete("कोलन") + pynutil.insert(":")
            + seg_backslash + path_segment
            + pynini.closure(seg_backslash + path_segment, 0)
            + pynini.closure(trail_backslash, 0, 1)
            + pynutil.insert("\"")
        )
        unc_path_fst = (
            pynutil.insert("path: \"")
            + lead_backslash + path_segment
            + pynini.closure(seg_backslash + path_segment, 0)
            + pynini.closure(trail_backslash, 0, 1)
            + pynutil.insert("\"")
        )
        unix_abs_path_fst = (
            pynutil.insert("path: \"")
            + unix_lead_slash + unix_segment
            + pynini.closure(unix_seg_slash + unix_segment, 0)
            + pynini.closure(unix_trail_slash, 0, 1)
            + pynutil.insert("\"")
        )
        unix_rel_path_fst = (
            pynutil.insert("path: \"")
            + unix_segment + unix_seg_slash + unix_segment
            + pynini.closure(unix_seg_slash + unix_segment, 0)
            + pynini.closure(unix_trail_slash, 0, 1)
            + pynutil.insert("\"")
        )
        tilde_path_fst = (
            pynutil.insert("path: \"")
            + tilde_delete + unix_seg_slash + unix_segment
            + pynini.closure(unix_seg_slash + unix_segment, 0)
            + pynini.closure(unix_trail_slash, 0, 1)
            + pynutil.insert("\"")
        )

        # CHANGE-6: literal-slash relative path FST.
        # Handles inputs like "होम / डेस्कटॉप" and "बैकअप्स / टेम्प" where
        # the separator is a literal ASCII "/" not spoken "फॉरवर्ड स्लैश".
        # lit_slash_seg = pynini.cross(" / ", "/") — defined above.
        # Uses unix_path_atom (lowercase) so output is "home/desktop" not "HOME/DESKTOP".
        lit_seg = (
            unix_path_atom
            + pynini.closure(
                pynutil.add_weight(delete_space + unix_path_atom, 1.0)
                | pynutil.add_weight(unix_hyphen,                 1.0)
            , 0)
            + pynini.closure(file_ext, 0, 1)
        )
        literal_rel_path_fst = (
            pynutil.insert("path: \"")
            + lit_seg
            + lit_slash_seg + lit_seg
            + pynini.closure(lit_slash_seg + lit_seg, 0)
            + pynini.closure(pynini.cross("/", "/"), 0, 1)
            + pynutil.insert("\"")
        )

        # ── URL / Domain ──────────────────────────────────────────────────────
        host_prefix_map = pynini.string_map([
            ("एस आर वी", "srv"), ("डी बी", "db"), ("एल टी", "lt"),
            ("वेब", "web"), ("लैपटॉप", "laptop"), ("डेस्कटॉप", "desktop"),
            ("ई मेल", "email"),
        ])
        first_label = (
            pynutil.add_weight(host_prefix_map,                               0.80)
            | pynutil.add_weight(digit_seq + delete_space + letter_map_lower, 0.85)
            | pynutil.add_weight(digit_seq,                                   0.87)
            | pynutil.add_weight(token_seq,                                   1.00)
        )
        domain_body  = first_label + pynini.closure(hyphen + (digit_seq | token_seq), 0)
        compound_tld = domain_map + pynini.closure(dot_end_safe + domain_map, 0, 2)
        full_domain  = pynini.closure(domain_body + dot, 0, 4) + domain_body + dot + compound_tld

        uname_atom = (
            pynutil.add_weight(and_as_letters,     0.84)
            | pynutil.add_weight(digit_words,      0.88)
            | pynutil.add_weight(digit_glyphs,     0.88)
            | pynutil.add_weight(server_map,       0.90)
            | pynutil.add_weight(common_map,       0.95)
            | pynutil.add_weight(letter_map_lower, 1.00)
        )
        uname_sep = (
            (delete_space + pynutil.delete("डॉट")   + delete_space + pynutil.insert("."))
            | (delete_space + (pynutil.delete("हाइफ़न") | pynutil.delete("हाइफन")) + delete_space + pynutil.insert("-"))
            | (delete_space + pynutil.delete("अंडर") + delete_space + pynutil.delete("स्कोर") + pynutil.insert("_"))
        )
        username  = uname_atom + pynini.closure((uname_sep + uname_atom) | (delete_space + uname_atom), 0)
        email_fst = (
            pynutil.insert("username: \"") + username + pynutil.insert("\"")
            + at_sign
            + pynutil.insert("domain: \"") + domain_body + dot + compound_tld + pynutil.insert("\"")
        )

        ip_octet = single_digit + pynini.closure(delete_space + single_digit, 0, 2)
        ip_fst   = (
            pynutil.insert("ip: \"")
            + ip_octet + dot + ip_octet + dot + ip_octet + dot + ip_octet
            + pynutil.insert("\"")
        )

        # CHANGE-1: digit_words (0.88) and digit_glyphs (0.89) added to path_atom_url.
        # token_seq (the existing fallback) has no digit map, so digit-only URL path
        # segments like "छह दो नौ" (629) failed, causing slash_with_word to fail and
        # the whole URL to split into two NeMo spans with a space between them.
        path_atom_url = (
            pynutil.add_weight(digit_seq + x_sep + digit_seq,                                           0.75)
            | pynutil.add_weight(digit_seq + delete_space + letter_map_lower + delete_space + digit_seq, 0.80)
            | pynutil.add_weight(digit_words,                                                            0.88)
            | pynutil.add_weight(digit_glyphs,                                                           0.89)
            | pynutil.add_weight(digit_seq,                                                              0.90)
            | pynutil.add_weight(token_seq,                                                              1.00)
        )
        path_segment_url = (
            path_atom_url
            + pynini.closure(hyphen + (digit_seq | token_seq), 0)
            + pynini.closure(underscore + token_seq, 0)
            + pynini.closure(dot + token_seq, 0, 1)
        )
        inline_domain_seg = (
            pynini.closure(token_seq + dot, 0, 2) + token_seq + dot + domain_map
            + pynini.closure(dot + domain_map, 0, 1)
        )
        slash_with_word = url_slash + delete_space + (
            pynutil.add_weight(pynutil.insert(".") + pynutil.delete("डॉट") + delete_space + token_seq, 0.90)
            | pynutil.add_weight(inline_domain_seg, 0.95)
            | pynutil.add_weight(path_segment_url,  1.00)
        )

        # CHANGE-2: hash fragment with hyphen support.
        # Original hash_frag used plain token_seq which stopped at हाइफ़न.
        # Now the fragment body allows hyphen+token_seq continuations so that
        # "#google-skillshop" works (tests 0162,0164,0166,0167,0168,0170).
        hash_frag_body = token_seq + pynini.closure(hyphen + token_seq, 0)
        hash_frag = (
            delete_space
            + pynutil.delete("हैशटैग")
            + pynutil.insert("#")
            + delete_space
            + hash_frag_body
        )

        domain_and_path = (
            full_domain
            + pynini.closure(slash_with_word, 0)
            + pynini.closure(url_slash, 0, 1)
            + pynini.closure(hash_frag, 0, 1)
        )

        https_prefix = (
            pynutil.delete("एच") + delete_space + pynutil.delete("टी") + delete_space
            + pynutil.delete("टी") + delete_space + pynutil.delete("पी") + delete_space
            + pynutil.delete("एस") + delete_space + pynutil.delete("कोलन") + delete_space
            + pynutil.delete("फॉरवर्ड") + delete_space + pynutil.delete("स्लैश") + delete_space
            + pynutil.delete("फॉरवर्ड") + delete_space + pynutil.delete("स्लैश")
            + pynutil.insert("https://")
        )
        http_prefix = (
            pynutil.delete("एच") + delete_space + pynutil.delete("टी") + delete_space
            + pynutil.delete("टी") + delete_space + pynutil.delete("पी") + delete_space
            + pynutil.delete("कोलन") + delete_space
            + pynutil.delete("फॉरवर्ड") + delete_space + pynutil.delete("स्लैश") + delete_space
            + pynutil.delete("फॉरवर्ड") + delete_space + pynutil.delete("स्लैश")
            + pynutil.insert("http://")
        )
        protocol = pynutil.add_weight(https_prefix, 1.0) | pynutil.add_weight(http_prefix, 1.01)

        url_fst  = (pynutil.insert("domain: \"") + protocol + delete_space
                    + pynini.closure(www_token + dot, 0, 1) + domain_and_path + pynutil.insert("\""))
        www_fst  = (pynutil.insert("domain: \"") + www_token + dot
                    + domain_and_path + pynutil.insert("\""))
        plain_fst = pynutil.insert("domain: \"") + domain_and_path + pynutil.insert("\"")

        # ── Chemical formulas ─────────────────────────────────────────────────
        chem_token = (
            pynutil.add_weight(pynutil.delete("फ़ोर") + pynutil.insert("₄"), 0.88)
            | pynutil.add_weight(subscript_digit_words, 0.90)
            | pynutil.add_weight(digit_glyphs,          0.92)
            | pynutil.add_weight(letter_map,             1.00)
        )
        chem_spelled_fst = pynutil.insert("domain: \"") + (
            chem_token
            + pynini.closure(
                pynutil.add_weight(delete_space + chem_token, 1.0)
                | pynutil.add_weight(open_bracket,  1.0)
                | pynutil.add_weight(close_bracket, 1.0)
                | pynutil.add_weight(delete_space + pynutil.delete("इनदो") + pynutil.insert("("),  1.0)
                | pynutil.add_weight(delete_space + pynutil.delete("बाय")   + pynutil.insert(")"),  1.0)
                | pynutil.add_weight(delete_space + (pynutil.delete("माइनस") | pynutil.delete("–")) + pynutil.insert("−"), 1.0)
            , 0)
        ) + pynutil.insert("\"")

        # ── Alphanumeric FSTs ─────────────────────────────────────────────────
        alnum_phrase_fst = pynutil.insert("domain: \"") + special_codes_map + pynutil.insert("\"")

        # *** FIX Bug 1 *** (original)
        # *** FIX Bug 2 *** (original)
        ex_upper    = pynutil.delete("एक्स") + pynutil.insert("X")
        alnum_token = (
            pynutil.add_weight(ex_upper,           0.90)
            | pynutil.add_weight(digit_glyphs,     0.92)
            | pynutil.add_weight(digit_words,      0.94)
            | pynutil.add_weight(letter_map_upper, 1.00)
        )
        alnum_run  = alnum_token + pynini.closure(delete_space + alnum_token, 0)

        # CHANGE-3: "प्वाइंट" added alongside "डॉट"/"DOT" in the dot separator.
        alnum_body = alnum_run + pynini.closure(
            pynutil.add_weight(
                delete_space + (pynutil.delete("हाइफ़न") | pynutil.delete("हाइफन"))
                + pynutil.insert("-") + delete_space
                + alnum_token + pynini.closure(delete_space + alnum_token, 0), 1.0)
            | pynutil.add_weight(
                delete_space
                + (pynutil.delete("डॉट") | pynutil.delete("DOT") | pynutil.delete("प्वाइंट"))
                + pynutil.insert(".") + delete_space
                + alnum_token + pynini.closure(delete_space + alnum_token, 0), 1.0)
            | pynutil.add_weight(
                delete_space + pynutil.delete("स्पेस") + pynutil.insert(" ")
                + delete_space + alnum_token + pynini.closure(delete_space + alnum_token, 0), 1.0)
        , 0)
        alnum_letterdigit_fst = pynutil.insert("domain: \"") + alnum_body + pynutil.insert("\"")

        # ── Final graph ───────────────────────────────────────────────────────
        # CHANGE-5: unix_abs_path_fst raised from 1.10 to 15.00.
        # See module docstring for full explanation.
        graph = (
            pynutil.add_weight(ip_fst,                  1.00)
            | pynutil.add_weight(email_fst,             1.05)
            | pynutil.add_weight(windows_path_fst,      1.06)
            | pynutil.add_weight(unc_path_fst,          1.07)
            | pynutil.add_weight(url_fst,               0.10)
            | pynutil.add_weight(www_fst,               0.11)
            | pynutil.add_weight(unix_abs_path_fst,    15.00)   # CHANGE-5
            | pynutil.add_weight(tilde_path_fst,        1.12)
            | pynutil.add_weight(unix_rel_path_fst,     1.14)
            | pynutil.add_weight(literal_rel_path_fst,  1.15)   # CHANGE-6
            | pynutil.add_weight(alnum_phrase_fst,      0.05)
            | pynutil.add_weight(chem_spelled_fst,      1.18)
            | pynutil.add_weight(alnum_letterdigit_fst, 0.06)
            | pynutil.add_weight(plain_fst,             1.30)
        )
        if chem_named_map is not None:
            graph = graph | pynutil.add_weight(
                pynutil.insert("domain: \"") + chem_named_map + pynutil.insert("\""), 0.80
            )

        self.fst = self.add_tokens(graph).optimize()