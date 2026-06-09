"""
Migration script to add scope_category and scope_confidence columns to the turns table.
Run this script to update the database schema.
"""
import asyncio
from sqlalchemy import text
from database import engine


async def add_scope_columns():
    """Add scope_category and scope_confidence columns to turns table."""
    async with engine.begin() as conn:
        # Check if columns already exist
        result = await conn.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'turns' AND column_name IN ('scope_category', 'scope_confidence')
        """))
        existing_columns = [row[0] for row in result]
        
        if 'scope_category' not in existing_columns:
            await conn.execute(text("""
                ALTER TABLE turns ADD COLUMN scope_category VARCHAR
            """))
            print("Added scope_category column")
        else:
            print("scope_category column already exists")
        
        if 'scope_confidence' not in existing_columns:
            await conn.execute(text("""
                ALTER TABLE turns ADD COLUMN scope_confidence FLOAT
            """))
            print("Added scope_confidence column")
        else:
            print("scope_confidence column already exists")
    
    print("Migration completed successfully")


if __name__ == "__main__":
    asyncio.run(add_scope_columns())
