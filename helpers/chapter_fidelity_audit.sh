#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

source_file="${1:-../ml_notes/linear_reg.tex}"
target_file="${2:-book/chapters/linear_regression.md}"
min_ratio="${3:-0.90}"

if [[ ! -f "$source_file" ]]; then
  echo "ERROR: source file not found: $source_file" >&2
  exit 1
fi

if [[ ! -f "$target_file" ]]; then
  echo "ERROR: target file not found: $target_file" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

fail=0

extract_source_headings() {
  perl -ne '
    if (/^\\subsection\{([^}]*)\}/) {
      print "## $1\n";
    } elsif (/^\\subsubsection\{([^}]*)\}/) {
      print "### $1\n";
    } elsif (/^\\paragraph\{([^}]*)\}/) {
      print "### $1\n";
    }
  ' "$1"
}

extract_target_headings() {
  perl -ne '
    if (/^(#{2,4})\s+(.*)$/) {
      print "$1 $2\n";
    }
  ' "$1"
}

extract_source_figures() {
  perl -ne '
    while (/\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}/g) {
      $f = $1;
      $f =~ s#.*/##;
      print "$f\n";
    }
  ' "$1" | sort -u
}

extract_target_figures() {
  perl -ne '
    if (/^\s*```{figure}\s+(.+?)\s*$/) {
      $f = $1;
      $f =~ s#.*/##;
      print "$f\n";
    }
    while (/!\[[^]]*\]\(([^)]+)\)/g) {
      $f = $1;
      $f =~ s#.*/##;
      print "$f\n";
    }
  ' "$1" | sort -u
}

extract_source_cites() {
  perl -ne '
    while (/\\cite\{([^}]*)\}/g) {
      @keys = split /,/, $1;
      for $k (@keys) {
        $k =~ s/\s+//g;
        print "$k\n" if $k ne "";
      }
    }
  ' "$1" | sort -u
}

extract_target_cites() {
  perl -ne '
    while (/\{cite\}`([^`]*)`/g) {
      @keys = split /,/, $1;
      for $k (@keys) {
        $k =~ s/\s+//g;
        print "$k\n" if $k ne "";
      }
    }
  ' "$1" | sort -u
}

count_source_display_math() {
  perl -0777 -ne '
    $t = $_;
    $count = 0;
    $count += () = $t =~ /\\begin\{equation\}/g;
    $count += () = $t =~ /(?<!\\)\\\[/g;
    $dollars = () = $t =~ /\$\$/g;
    $count += int($dollars / 2);
    print "$count\n";
  ' "$1"
}

count_target_display_math() {
  perl -0777 -ne '
    $t = $_;
    $count = 0;
    $count += () = $t =~ /^```{math}/mg;
    $dollars = () = $t =~ /\$\$/g;
    $count += int($dollars / 2);
    print "$count\n";
  ' "$1"
}

echo "== Heading Hierarchy Parity =="
extract_source_headings "$source_file" > "$tmpdir/source_headings.txt"
extract_target_headings "$target_file" > "$tmpdir/target_headings.txt"
if diff -u "$tmpdir/source_headings.txt" "$tmpdir/target_headings.txt"; then
  echo "PASS: heading hierarchy matches."
else
  echo "FAIL: heading hierarchy mismatch."
  fail=1
fi
echo

echo "== Display-Math Count Parity =="
source_math_count="$(count_source_display_math "$source_file" | tr -d ' ')"
target_math_count="$(count_target_display_math "$target_file" | tr -d ' ')"
echo "source display-math count: $source_math_count"
echo "target display-math count: $target_math_count"
if [[ "$source_math_count" == "$target_math_count" ]]; then
  echo "PASS: display-math counts match."
else
  echo "FAIL: display-math counts differ."
  fail=1
fi
echo

echo "== Figure Reference Parity =="
extract_source_figures "$source_file" > "$tmpdir/source_figures.txt"
extract_target_figures "$target_file" > "$tmpdir/target_figures.txt"
if diff -u "$tmpdir/source_figures.txt" "$tmpdir/target_figures.txt"; then
  echo "PASS: figure references match."
else
  echo "FAIL: figure references mismatch."
  fail=1
fi
echo

echo "== Citation Key Parity =="
extract_source_cites "$source_file" > "$tmpdir/source_cites.txt"
extract_target_cites "$target_file" > "$tmpdir/target_cites.txt"
if diff -u "$tmpdir/source_cites.txt" "$tmpdir/target_cites.txt"; then
  echo "PASS: citation keys match."
else
  echo "FAIL: citation keys mismatch."
  fail=1
fi
echo

echo "== Normalized Text-Length Parity =="
if command -v pandoc >/dev/null 2>&1; then
  source_words="$(pandoc -f latex -t plain "$source_file" 2>/dev/null | wc -w | tr -d ' ')"
  target_words="$(pandoc -f gfm -t plain "$target_file" 2>/dev/null | wc -w | tr -d ' ')"
else
  source_words="$(perl -pe 's/\\[a-zA-Z@]+(\{[^}]*\})?//g; s/[\{\}\\$]/ /g' "$source_file" | wc -w | tr -d ' ')"
  target_words="$(perl -pe 's/`{1,3}[^`]*`{1,3}//g; s/[#*_`:{}\[\]\(\)]/ /g' "$target_file" | wc -w | tr -d ' ')"
fi
ratio="$(awk -v s="$source_words" -v t="$target_words" 'BEGIN { if (s == 0) print 0; else printf "%.4f", t/s }')"
echo "source words: $source_words"
echo "target words: $target_words"
echo "ratio (target/source): $ratio"
if awk -v r="$ratio" -v min="$min_ratio" 'BEGIN { exit !(r >= min) }'; then
  echo "PASS: normalized text-length ratio meets threshold >= $min_ratio."
else
  echo "FAIL: normalized text-length ratio below threshold >= $min_ratio."
  fail=1
fi
echo

if [[ "$fail" -ne 0 ]]; then
  echo "Chapter fidelity audit: FAIL"
  exit 1
fi

echo "Chapter fidelity audit: PASS"
