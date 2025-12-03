import re


class LocationExtractor:
    def _extract_address(self, text: str) -> str:
        pattern = re.compile(
            r"\b(\d{1,5}\s+[A-Za-z][A-Za-z\s]*\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr))\b",
            flags=re.IGNORECASE,
        )
        m = pattern.search(text)
        if m:
            return m.group(1).strip()
        return ""

    def extract(self, text: str) -> str:
        address = self._extract_address(text)
        if address:
            return address

        lowered = text.lower()

        if "mall" in lowered:
            return "the mall"
        if "station" in lowered:
            return "the station"
        if "airport" in lowered:
            return "the airport"

        patterns = [
            r"near ([^.,;]+)",
            r"at ([^.,;]+)",
            r"in ([^.,;]+)",  # very rough, improves later
        ]
        candidate = ""
        for pattern in patterns:
            m = re.search(pattern, lowered)
            if m:
                candidate = m.group(0)
                break

        if not candidate:
            return ""

        geo_tokens = [
            "street",
            "st ",
            "st.",
            "avenue",
            "ave",
            "road",
            "rd",
            "boulevard",
            "blvd",
            "lane",
            "ln",
            "drive",
            "dr",
            "downtown",
            "city",
            "center",
            "centre",
            "plaza",
            "square",
            "park",
            "town",
            "mall",
            "station",
            "airport",
        ]

        if any(tok in candidate for tok in geo_tokens):
            return candidate.strip()

        return ""
