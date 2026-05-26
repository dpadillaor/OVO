#!/usr/bin/env python3
"""
Load and cherry-pick from Obsidian daily memory logs.

Usage:
  uv run .claude/scripts/load-daily-memory.py yesterday --sections "Actions completed" --tags fusion --summary brief
  uv run .claude/scripts/load-daily-memory.py today --sessions-summary
  uv run .claude/scripts/load-daily-memory.py 3-days-ago --open-tasks
"""

import sys
import argparse
import re
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class Session:
    """Represents a single session's data."""
    timestamp: str
    title: str
    sections: Dict[str, str]
    
    def __repr__(self) -> str:
        return f"<Session {self.timestamp} | {self.title}>"


class DailyMemoryLoader:
    """Parse and filter Obsidian daily memory logs."""
    
    VAULT_DAILY = Path.home() / "Documentos" / "Obsidian Vault" / "daily"
    SESSION_DELIMITER = r"^> \[!abstract\]-"
    SECTION_PATTERN = r"^\*\*([^*]+)\*\*:?\s*$"
    TAG_PATTERN = r"#(\w+)"
    CANONICAL_SECTIONS = {
        "What happened",
        "Decisions & reasoning",
        "Actions completed",
        "Open tasks",
        "Connected to",
    }
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.sessions: List[Session] = []
        self.parse()
    
    def parse(self) -> None:
        """Parse file into sessions."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by session delimiter
        session_blocks = re.split(self.SESSION_DELIMITER, content, flags=re.MULTILINE)
        
        for block in session_blocks[1:]:  # Skip first empty split
            session = self._parse_session_block(block)
            if session:
                self.sessions.append(session)
    
    def _parse_session_block(self, block: str) -> Optional[Session]:
        """Parse a single session block."""
        lines = block.split('\n')
        
        # Extract timestamp from first line
        # Format: "- Session End · 16:43 · `uuid`" or "- PreCompact · 22:14 · `uuid`"
        first_line = lines[0].strip() if lines else ""
        timestamp = self._extract_timestamp(first_line)
        
        # Extract title (usually the next **bold** line)
        title = self._extract_title(lines)
        
        # Parse sections
        sections = self._extract_sections('\n'.join(lines))
        
        if sections:
            return Session(timestamp=timestamp, title=title, sections=sections)
        return None
    
    def _extract_timestamp(self, line: str) -> str:
        """Extract time from session header."""
        match = re.search(r'·\s*(\d{1,2}:\d{2})\s*·', line)
        if match:
            return match.group(1)
        return "unknown"
    
    def _extract_title(self, lines: List[str]) -> str:
        """Extract session title (first **bold** line or first non-empty line)."""
        for line in lines:
            if line.startswith('**') and line.endswith('**'):
                return line.strip('*').strip(':').strip()
            if line.strip() and not line.startswith('>') and not line.startswith('-'):
                return line.strip()[:80]
        return "Untitled"
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Parse sections: **SectionName**: content."""
        sections = {}
        current_section = None
        current_content = []
        
        for line in text.split('\n'):
            # Match section header
            match = re.match(self.SECTION_PATTERN, line)
            if match and match.group(1).strip() in self.CANONICAL_SECTIONS:
                # Save previous section
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                current_section = match.group(1).strip()
                current_content = []
            elif current_section:
                current_content.append(line)
        
        # Save last section
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def extract_tags(self, text: str) -> set:
        """Extract all #tags from text."""
        return set(re.findall(self.TAG_PATTERN, text))
    
    def filter_sessions(
        self,
        sections_filter: Optional[List[str]] = None,
        tags_filter: Optional[List[str]] = None,
        sessions_only: Optional[List[int]] = None
    ) -> List[Session]:
        """Filter sessions by criteria."""
        result = []
        
        for idx, session in enumerate(self.sessions):
            # Filter by session index if specified
            if sessions_only and idx not in sessions_only:
                continue
            
            # Apply section/tag filters
            filtered_sections = self._filter_session_sections(
                session, sections_filter, tags_filter
            )
            
            if filtered_sections or not (sections_filter or tags_filter):
                # Create new session with filtered content
                filtered_session = Session(
                    timestamp=session.timestamp,
                    title=session.title,
                    sections=filtered_sections if filtered_sections else session.sections
                )
                result.append(filtered_session)
        
        return result
    
    def _filter_session_sections(
        self,
        session: Session,
        sections_filter: Optional[List[str]] = None,
        tags_filter: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """Filter a session's sections."""
        filtered = {}
        
        for section_name, section_content in session.sections.items():
            # Apply section name filter
            if sections_filter and section_name not in sections_filter:
                continue
            
            # Apply tag filter
            if tags_filter:
                found_tags = self.extract_tags(section_content)
                if not any(tag in found_tags for tag in tags_filter):
                    continue
            
            filtered[section_name] = section_content
        
        return filtered
    
    def format_output(
        self,
        sessions: List[Session],
        summary_level: str = "full",
        show_timestamps: bool = True
    ) -> str:
        """Format sessions for output."""
        lines = []
        
        for session in sessions:
            if show_timestamps:
                header = f"📅 {session.timestamp} | {session.title}"
                lines.append(header)
                lines.append("-" * 70)
            else:
                lines.append(f"▬ {session.title}")
            
            # Format sections in canonical order
            section_order = [
                "What happened", "Decisions & reasoning", "Actions completed",
                "Open tasks", "Connected to"
            ]
            
            for section_name in section_order:
                if section_name in session.sections:
                    content = session.sections[section_name]
                    formatted = self._format_section(section_name, content, summary_level)
                    if formatted:
                        lines.append(formatted)
                        lines.append("")
            
            lines.append("=" * 70)
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_section(self, title: str, content: str, level: str) -> Optional[str]:
        """Format a single section."""
        if not content.strip():
            return None
        
        if level == "brief":
            first_line = content.split('\n')[0].strip()
            return f"**{title}**: {first_line}"
        elif level == "outline":
            bullets = [line for line in content.split('\n') if line.strip().startswith('-')]
            if bullets:
                return f"**{title}**\n" + "\n".join(bullets)
            return f"**{title}**: (no bullets)"
        else:  # full
            return f"**{title}**\n{content}"


def resolve_date(spec: str) -> Path:
    """Convert date spec to file path."""
    today = datetime.now().date()
    
    if spec.endswith('.md'):
        spec = spec[:-3]
    
    # Absolute date: YYYY-MM-DD
    if spec.count('-') == 2:
        try:
            target = datetime.strptime(spec, '%Y-%m-%d').date()
        except ValueError:
            raise ValueError(f"Invalid date format: {spec}")
    # Relative dates
    elif spec == "today":
        target = today
    elif spec == "yesterday":
        target = today - timedelta(days=1)
    elif spec == "week-ago":
        target = today - timedelta(days=7)
    elif spec == "month-ago":
        target = today - timedelta(days=30)
    elif spec.endswith('-days-ago'):
        try:
            days = int(spec.split('-')[0])
            target = today - timedelta(days=days)
        except ValueError:
            raise ValueError(f"Invalid days spec: {spec}")
    else:
        raise ValueError(f"Unknown date spec: {spec}")
    
    file_path = DailyMemoryLoader.VAULT_DAILY / f"{target.strftime('%Y-%m-%d')}.md"
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        available = sorted(DailyMemoryLoader.VAULT_DAILY.glob("*.md"), reverse=True)[:5]
        if available:
            print("\n📋 Closest available dates:")
            for f in available:
                print(f"   • {f.stem}")
        sys.exit(1)
    
    return file_path


def main():
    parser = argparse.ArgumentParser(
        description="Load and cherry-pick from Obsidian daily memory logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s today
  %(prog)s yesterday --sections "Actions completed" --tags fusion
  %(prog)s 3-days-ago --summary brief --sessions-summary
  %(prog)s today --open-tasks
        """
    )
    
    parser.add_argument(
        'date',
        help='Date spec: today, yesterday, 3-days-ago, week-ago, month-ago, or YYYY-MM-DD'
    )
    parser.add_argument(
        '--sections',
        help='Comma-separated section names (e.g. "What happened, Actions completed")',
        type=lambda s: [x.strip() for x in s.split(',')]
    )
    parser.add_argument(
        '--tags',
        help='Filter by tags (space-separated)',
        nargs='*'
    )
    parser.add_argument(
        '--summary',
        choices=['full', 'brief', 'outline'],
        default='full',
        help='Output format (default: full)'
    )
    parser.add_argument(
        '--sessions-summary',
        action='store_true',
        help='Show summary of all sessions without timestamps'
    )
    parser.add_argument(
        '--open-tasks',
        action='store_true',
        help='Extract only "Open tasks" section from all sessions'
    )
    parser.add_argument(
        '--sessions',
        type=lambda s: [int(x) for x in s.split(',')],
        help='Specific session indices (0-based, comma-separated)'
    )
    
    args = parser.parse_args()
    
    try:
        file_path = resolve_date(args.date)
        loader = DailyMemoryLoader(file_path)
        
        # Determine filtering strategy
        if args.open_tasks:
            sections_filter = ["Open tasks"]
            show_timestamps = True
        elif args.sessions_summary:
            sections_filter = None
            show_timestamps = False
        else:
            sections_filter = args.sections
            show_timestamps = True
        
        # Filter sessions
        filtered_sessions = loader.filter_sessions(
            sections_filter=sections_filter,
            tags_filter=args.tags if args.tags else None,
            sessions_only=args.sessions
        )
        
        if not filtered_sessions:
            print("✓ No sessions match your criteria.")
            sys.exit(0)
        
        # Format and output
        output = loader.format_output(
            filtered_sessions,
            summary_level=args.summary,
            show_timestamps=show_timestamps
        )
        print(output)
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
