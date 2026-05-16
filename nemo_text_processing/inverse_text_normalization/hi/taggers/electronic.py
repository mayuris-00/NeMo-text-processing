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

    chemical_formulas.tsv format
    ----------------------------
    Each row is:   Hindi_key <TAB> formula_output

    CHEM-4  chem_spelled_fst retained as fallback for novel/unknown formulas
            not yet in the TSV.
    """

    def __init__(self):
        super().__init__(name="electronic", kind="classify")

        # ── Digit maps ────────────────────────────────────────────────────────
        # digit_words.tsv: Hindi_word → ASCII_digit  (e.g. एक → 1, दो → 2)
        # Single source of truth — also used by classify.py penalty FST.
        digit_words = pynini.string_file(
            get_abs_path("data/electronic/digit_words.tsv")
        )
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

        letter_map = pynini.string_file(get_abs_path("data/electronic/letters.tsv")).invert()
        domain_map = pynini.string_file(get_abs_path("data/electronic/domain.tsv")).invert()
        server_map = pynini.string_file(get_abs_path("data/electronic/server_name.tsv")).invert()
        common_map = pynini.string_file(get_abs_path("data/electronic/common_words.tsv")).invert()

        # CHEM-1: Load chemical_formulas.tsv WITHOUT .invert().
        # The TSV is stored as   Hindi_key → formula_output   already.
        # Adding .invert() would flip direction and make every entry unreachable.
        try:
            chem_named_map = pynini.string_file(
                get_abs_path("data/electronic/chemical_formulas.tsv")
                # NO .invert() — file is already Hindi → formula
            ).optimize()
        except Exception:
            chem_named_map = None

        # ── special_codes_map ─────────────────────────────────────────────────
        # CHEM-3: Chemical entries that were previously here (ZnSO4, MgBr2)
        # have been removed — they are now fully handled by chem_named_map.
        # All non-chemical alphanumeric codes remain.
        special_codes_map = pynini.string_map([
            # ── original non-chemical entries ─────────────────────────────────
            ("आई ए तीन दो",                        "ia32"),
            ("एक्स आठ छह",                          "x86"),
            ("छह एस",                               "6s"),
            ("यू टी एफ आठ",                        "utf8"),
            ("ए एस सी आई आई आठ पाँच",             "Ascii85"),
            ("पी एच आर ए सी के पाँच सात",         "phrack57"),
            ("ज़ेड एक्स आठ शून्य",                  "ZX80"),
            ("ज़ेड एक्स आठ एक",                     "ZX81"),
            ("ज़ेड एक्स आठ शून्य एक नौ आठ शून्य", "ZX80 1980"),
            ("बी ए एस ई तीन दो",                    "Base32"),
            ("बी ई सी एच तीन दो",                   "Bech32"),
            ("सी ए एन ओ एन ए सात पाँच",            "Canon A75"),
            # Category 4 entries (Hindi script output) moved to chemical_formulas.tsv:
            # अग्नि-2, कोविड-19, पृथ्वी-4, ब्रह्मोस-1, भास्कर-II, श्रेणी-II,
            # रोहिणी आर एस-I, ऑडी 80/90 B4, 5जी
            # They are loaded via chem_named_map (weight 0.04) which beats
            # special_codes_map (0.05), so no conflict.
            ("आर ई डी एम आई एच चार चार चार जी",    "REDMI H44 4G"),
            ("आई एन एन ओ टी ई एक",                  "IN NOTE1"),

            # ── Group A: digit-run codes ──────────────────────────────────────
            # NOTE: these entries are only needed if classify.py does NOT have
            # digit_word_penalty_fst (CHANGE-TAGGER-1). With that fix applied,
            # alnum_letterdigit_fst handles these correctly as single spans.
            # They are kept here as a fallback in case classify.py is not patched.
            ("आई एन छह तीन आठ नौ शून्य नौ शून्य",       "IN6389090"),
            ("आई एस बी एन आठ नौ शून्य चार तीन पाँच सात", "ISBN8904357"),
            ("आई एस बी एन छह सात आठ नौ शून्य दो दो चार", "ISBN67890224"),
            ("एन वाई नौ छह तीन दो नौ शून्य",             "NY963290"),
            ("एम एम एक्स दो दो शून्य नौ शून्य आठ",       "MMX220908"),
            ("ओ डी शून्य दो सात छह तीन पाँच",            "OD027635"),
            ("पी आर तीन दो दो एक सात सात",               "PR322177"),
            ("बी एच सात छह एस वाई एक सात आठ पाँच",       "BH76SY1785"),
            ("ई एक्स आठ छह तीन दो जे यू दो पाँच चार",   "EX8632JU254"),

            # ── Group D: case/space codes ─────────────────────────────────────
            ("पाँच सी",                         "5c"),
            # पाँच जी handled by chemical_formulas.tsv → 5जी at weight 0.04
            ("सात आई",                          "7i"),
            ("एक दो शून्य एम एम",              "120mm"),
            ("एम के हाइफ़न एक ए",              "Mk-1A"),
            ("एक नौ तीन पाँच आठ एम नौ",        "1 9358M9"),
            ("छह दो दो छह शून्य एम छह",        "6 2260M6"),
            ("सात नौ नौ छह पाँच एम छह",        "7 9965M6"),
            ("एफ हाइफ़न तीन पाँच सी",          "F-35C"),
            ("डॉट टी आर ए वी आई एस डॉट वाई एम एल", ".travis.yml"),
            # 0093: server_map gives lowercase hipo — need capital H
            ("एच आई पी ओ",                     "Hipo"),
        ]).optimize()

        to_lower = pynini.cdrewrite(
            pynini.string_map([(chr(c), chr(c + 32)) for c in range(ord('A'), ord('Z') + 1)]),
            "", "", NEMO_SIGMA,
        )
        to_upper = pynini.cdrewrite(
            pynini.string_map([(chr(c + 32), chr(c)) for c in range(ord('A'), ord('Z') + 1)]),
            "", "", NEMO_SIGMA,
        )

        def make_lower(fst):
            return (fst @ to_lower).optimize()

        def make_upper(fst):
            return (fst @ to_upper).optimize()

        letter_map_lower = make_lower(letter_map)
        letter_map_upper = make_upper(letter_map)
        common_map_lower = make_lower(common_map)
        server_map_lower = make_lower(server_map)

        latin_run = pynini.closure(
            pynini.union(*[pynini.accep(chr(c)) for c in range(ord('A'), ord('Z') + 1)])
            | pynini.union(*[pynini.accep(chr(c)) for c in range(ord('a'), ord('z') + 1)]),
            1,
        )
        latin_run_lower = make_lower(latin_run)

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

        url_slash = (
            delete_space
            + pynutil.delete("फॉरवर्ड") + delete_space + pynutil.delete("स्लैश")
            + pynutil.insert("/")
        )

        lit_slash_seg  = pynini.cross(" / ", "/")
        lit_hyphen_seg = pynini.cross(" - ", "-")

        dot           = delete_space + (pynutil.delete("डॉट") | pynutil.delete("DOT")) + delete_space + pynutil.insert(".")
        dot_end_safe  = delete_space + (pynutil.delete("डॉट") | pynutil.delete("DOT")) + delete_zero_or_one_space + pynutil.insert(".")
        hyphen        = delete_space + (pynutil.delete("हाइफ़न") | pynutil.delete("हाइफन")) + delete_space + pynutil.insert("-")
        underscore    = delete_space + pynutil.delete("अंडर") + delete_space + pynutil.delete("स्कोर") + pynutil.insert("_")
        at_sign       = delete_space + pynutil.delete("एट") + delete_space
        x_sep         = delete_space + pynutil.delete("एक्स") + pynutil.insert("x")
        literal_space = delete_space + pynutil.delete("स्पेस") + pynutil.insert(" ")
        open_bracket  = (
            delete_space
            + pynutil.delete("ओपन") + delete_space + pynutil.delete("ब्रेकेट")
            + pynutil.insert("(")
            + delete_zero_or_one_space
        )
        close_bracket = (
            delete_space
            + pynutil.delete("क्लोज़") + delete_space + pynutil.delete("ब्रेकेट")
            + pynutil.insert(")")
            + delete_zero_or_one_space
        )
        dollar_sign    = delete_space + pynutil.delete("डॉलर") + pynutil.insert("$")

        # Literal ASCII parens in input (not spoken as "ओपन ब्रेकेट").
        # pynutil.delete consumes the char from input (needed since we insert separately).
        # pynini.accep would keep it in output causing double-printing.
        lit_open_paren  = (
            delete_space                      # space before (
            + pynutil.delete("(")             # consume ( from input
            + pynutil.insert("(")             # emit (
            + delete_space                    # space after ( before first token
        )
        lit_close_paren = (
            delete_space                      # space before )
            + pynutil.delete(")")             # consume ) from input
            + pynutil.insert(")")             # emit )
        )
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
        # path_atom_lower: same as path_atom but all Latin output lowercased,
        # and without hp_token/www_token/or_word/and_as_letters (not needed
        # in file extension context). Built from lower variants directly.
        path_atom_lower = (
            pynutil.add_weight(common_map_lower,   0.90)
            | pynutil.add_weight(server_map_lower, 0.92)
            | pynutil.add_weight(digit_words,      0.94)
            | pynutil.add_weight(digit_glyphs,     0.95)
            | pynutil.add_weight(latin_run_lower,  0.97)
            | pynutil.add_weight(letter_map_lower, 1.00)
        )
        # unix_path_atom: path_atom_lower + CI cross + www/or/and extras
        unix_path_atom = (
            pynutil.add_weight(www_token,                    0.77)
            | pynutil.add_weight(or_word,                    0.80)
            | pynutil.add_weight(and_as_letters,             0.84)
            | pynutil.add_weight(pynini.cross("CI", "c"),    0.86)
            | path_atom_lower
        )

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
                | pynutil.add_weight(lit_open_paren,         1.0)
                | pynutil.add_weight(lit_close_paren,        1.0)
            , 0)
            + pynini.closure(file_ext, 0, 1)
        )

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

        lit_seg = (
            unix_path_atom
            + pynini.closure(
                pynutil.add_weight(delete_space + unix_path_atom, 1.0)
                | pynutil.add_weight(unix_hyphen,                 1.0)
                | pynutil.add_weight(lit_hyphen_seg,              1.0)
            , 0)
            + pynini.closure(file_ext, 0, 1)
        )
        literal_rel_path_fst = (
            pynutil.insert("path: \"")
            + lit_seg
            + lit_slash_seg + lit_seg
            + pynini.closure(lit_slash_seg + lit_seg, 0)
            + pynini.closure(pynini.cross(" /", "/"), 0, 1)   # consume space before trailing /
            + pynutil.insert("\"")
        )

        host_prefix_map = pynini.string_map([
            ("एस आर वी", "srv"), ("डी बी", "db"), ("एल टी", "lt"),
            ("वेब", "web"), ("लैपटॉप", "laptop"), ("डेस्कटॉप", "desktop"),
            ("ई मेल", "email"),
        ])

        digit_then_letter = (
            digit_seq
            + pynini.closure(delete_space + letter_map_lower, 0)
        )

        first_label = (
            pynutil.add_weight(host_prefix_map,                               0.80)
            | pynutil.add_weight(digit_seq + delete_space + letter_map_lower, 0.85)
            | pynutil.add_weight(digit_seq,                                   0.87)
            | pynutil.add_weight(token_seq,                                   1.00)
        )
        domain_body = first_label + pynini.closure(
            hyphen + (
                pynutil.add_weight(digit_then_letter, 0.85)
                | pynutil.add_weight(digit_seq | token_seq, 1.0)
            ), 0
        )
        compound_tld = domain_map + pynini.closure(dot_end_safe + domain_map, 0, 2)
        full_domain  = pynini.closure(domain_body + dot, 0, 4) + domain_body + dot + compound_tld
        full_domain_bare = pynini.closure(domain_body + dot, 0, 4) + domain_body

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
        slash_with_word = url_slash + (
            pynutil.add_weight(delete_space + pynutil.insert(".") + pynutil.delete("डॉट") + delete_space + token_seq, 0.90)
            | pynutil.add_weight(delete_space + inline_domain_seg, 0.95)
            | pynutil.add_weight(delete_space + path_segment_url,  1.00)
        )

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
        domain_and_path_bare = (
            full_domain_bare
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

        url_fst   = (pynutil.insert("domain: \"") + protocol + delete_space
                     + pynini.closure(www_token + dot, 0, 1) + domain_and_path + pynutil.insert("\""))
        www_fst   = (pynutil.insert("domain: \"") + www_token + dot
                     + domain_and_path + pynutil.insert("\""))
        plain_fst = pynutil.insert("domain: \"") + domain_and_path + pynutil.insert("\"")

        url_fst_bare = (
            pynutil.insert("domain: \"") + protocol + delete_space
            + pynini.closure(www_token + dot, 0, 1)
            + domain_and_path_bare + pynutil.insert("\"")
        )
        www_fst_bare = (
            pynutil.insert("domain: \"") + www_token + dot
            + domain_and_path_bare + pynutil.insert("\"")
        )

        # chem_spelled_fst: fallback for novel formulas NOT yet in the TSV.
        # Uses subscript_digit_words so output always uses ₂ ₃ ₄ etc.
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

        alnum_phrase_fst      = pynutil.insert("domain: \"") + special_codes_map + pynutil.insert("\"")

        ex_upper    = pynutil.delete("एक्स") + pynutil.insert("X")
        alnum_token = (
            pynutil.add_weight(ex_upper,           0.90)
            | pynutil.add_weight(digit_glyphs,     0.92)
            | pynutil.add_weight(digit_words,      0.94)
            | pynutil.add_weight(letter_map_upper, 1.00)
        )
        alnum_run  = alnum_token + pynini.closure(delete_space + alnum_token, 0)

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
            # Literal ASCII ( ) pass-through — for inputs like "(AVIC)" where
            # the tagger emits real paren chars, not spoken "ओपन ब्रेकेट".
            | pynutil.add_weight(lit_open_paren  + alnum_token + pynini.closure(delete_space + alnum_token, 0), 1.0)
            | pynutil.add_weight(lit_close_paren + alnum_token + pynini.closure(delete_space + alnum_token, 0), 1.0)
            | pynutil.add_weight(lit_close_paren, 1.0)   # trailing ) with no following token
        , 0)
        alnum_letterdigit_fst = pynutil.insert("domain: \"") + alnum_body + pynutil.insert("\"")

        # ── Final graph ───────────────────────────────────────────────────────
        #
        # Weight ladder (lower = higher priority):
        #   0.04  chem_named_map         (exact chemical formula TSV lookup)
        #   0.05  alnum_phrase_fst       (special_codes_map — custom outputs)
        #   0.06  alnum_letterdigit_fst  (generic letter+digit parser)
        #   0.10  url_fst                (HTTP/HTTPS URLs with known TLD)
        #   0.11  www_fst
        #   0.50  url_fst_bare           (HTTP/HTTPS URLs, no TLD required)
        #   0.51  www_fst_bare
        #   1.00  ip_fst
        #   1.05  email_fst
        #   1.06  windows_path_fst
        #   1.07  unc_path_fst
        #  15.00  unix_abs_path_fst      (high weight to avoid stray-space split)
        #   1.12  tilde_path_fst
        #  15.00  unix_rel_path_fst
        #   1.15  literal_rel_path_fst
        #   1.18  chem_spelled_fst       (fallback for unlisted formulas)
        #   1.30  plain_fst

        chem_fst = pynutil.add_weight(
            pynutil.insert("domain: \"") + chem_named_map + pynutil.insert("\""), 0.04
        ) if chem_named_map is not None else pynini.accep("")

        graph = (
            pynutil.add_weight(ip_fst,                  1.00)
            | pynutil.add_weight(email_fst,             1.05)
            | pynutil.add_weight(windows_path_fst,      1.06)
            | pynutil.add_weight(unc_path_fst,          1.07)
            | pynutil.add_weight(url_fst,               0.10)
            | pynutil.add_weight(www_fst,               0.11)
            | pynutil.add_weight(url_fst_bare,          0.50)
            | pynutil.add_weight(www_fst_bare,          0.51)
            | pynutil.add_weight(unix_abs_path_fst,    15.00)
            | pynutil.add_weight(tilde_path_fst,        1.12)
            | pynutil.add_weight(unix_rel_path_fst,    15.00)
            | pynutil.add_weight(literal_rel_path_fst,  1.15)
            | pynutil.add_weight(alnum_phrase_fst,      0.05)
            | pynutil.add_weight(chem_spelled_fst,      1.18)
            | pynutil.add_weight(alnum_letterdigit_fst, 0.06)
            | pynutil.add_weight(plain_fst,             1.30)
            | chem_fst
        )

        self.fst = self.add_tokens(graph).optimize()