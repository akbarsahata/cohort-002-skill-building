Right now, your algorithm can handle emails just fine - but only if they're reasonably short. Longer emails can become problematic, especially when you're trying to process them with an LLM.

The solution is to add chunking. Instead of sending the entire email at once, you'll break it into smaller, manageable pieces. This is where a structure-aware text splitter comes in handy.

By using a structure-aware approach, you can split your text intelligently - respecting paragraph boundaries rather than cutting through the middle of sentences or ideas. This keeps your processing clean and meaningful.

## Abstracting The Embeddings Cache

<!-- VIDEO -->

We're going to abstract the embeddings cache into a separate module. This makes it easier to work with when we start chunking emails, since we'll have multiple embeddings per email.

Recommendation: cherry-pick, and read the code carefully. This commit is mostly busywork that sets up the rest of this exercise.

### Steps To Complete

#### Create the `embeddings.ts` file

- [ ] Create a new file at `src/app/embeddings.ts`. We're extracting existing cache logic from `search.ts` into a reusable module.

- [ ] Add constants for the cache directory path and the embedding model key:

<Spoiler>

```typescript
// src/app/embeddings.ts
import path from 'path';

const CACHE_DIR = path.join(process.cwd(), 'data', 'embeddings');
const CACHE_KEY = 'google-text-embedding-004';
```

</Spoiler>

- [ ] Instead of using email IDs for cache file names, create a function that hashes the content. This way, the same content always maps to the same cache file, even if it appears in multiple emails after chunking.

<Spoiler>

```typescript
// src/app/embeddings.ts
import crypto from 'crypto';

const getEmbeddingFilePath = (content: string) => {
  const hash = crypto
    .createHash('sha256')
    .update(content)
    .digest('hex')
    .slice(0, 10);

  return path.join(CACHE_DIR, `${CACHE_KEY}-${hash}.json`);
};
```

</Spoiler>

- [ ] Create an exported function that ensures the embeddings cache directory exists:

<Spoiler>

```typescript
// src/app/embeddings.ts
import fs from 'fs/promises';

export const ensureEmbeddingsCacheDirectory = async () => {
  await fs.mkdir(CACHE_DIR, { recursive: true });
};
```

</Spoiler>

- [ ] Create an exported function that attempts to read a cached embedding. It should return the embedding if found, or `null` if not:

<Spoiler>

```typescript
// src/app/embeddings.ts

export const getCachedEmbedding = async (
  content: string,
): Promise<number[] | null> => {
  const filePath = getEmbeddingFilePath(content);
  try {
    const cached = await fs.readFile(filePath, 'utf-8');
    return JSON.parse(cached);
  } catch {
    return null;
  }
};
```

</Spoiler>

- [ ] Create an exported function that writes an embedding to the cache for a given content string:

<Spoiler>

```typescript
// src/app/embeddings.ts

export const writeEmbeddingToCache = async (
  content: string,
  embedding: number[],
) => {
  const filePath = getEmbeddingFilePath(content);
  await fs.writeFile(filePath, JSON.stringify(embedding));
};
```

</Spoiler>

#### Update `search.ts` to use the new module

- [ ] Import the new functions from `embeddings.ts`:

```typescript
import {
  ensureEmbeddingsCacheDirectory,
  getCachedEmbedding,
  writeEmbeddingToCache,
} from './embeddings';
```

- [ ] Remove the old `CACHE_DIR`, `CACHE_KEY`, and `getEmbeddingFilePath` definitions from `search.ts`.

- [ ] Add a helper function that converts an email to text format. This ensures consistent formatting since we're now hashing the text instead of using email IDs:

<Spoiler>

```typescript
// src/app/search.ts

export const emailToText = (email: Email) =>
  `${email.subject} ${email.body}`;
```

</Spoiler>

- [ ] Update the `searchWithBM25` function to use `emailToText`:

<Spoiler>

```typescript
export async function searchWithBM25(
  keywords: string[],
  emails: Email[],
) {
  const corpus = emails.map((email) => emailToText(email));

  const scores: number[] = (BM25 as any)(corpus, keywords);

  return scores
    .map((score, idx) => ({ score, email: emails[idx] }))
    .sort((a, b) => b.score - a.score);
}
```

</Spoiler>

#### Update `loadOrGenerateEmbeddings`

- [ ] Replace `await fs.mkdir(CACHE_DIR, { recursive: true })` with a call to the new helper function at the start of the function:

<Spoiler>

```typescript
export async function loadOrGenerateEmbeddings(
  emails: Email[],
): Promise<{ id: string; embedding: number[] }[]> {
  await ensureEmbeddingsCacheDirectory();

  // ... rest of function
}
```

</Spoiler>

- [ ] In the cache checking loop, replace the manual `fs.readFile` logic with `getCachedEmbedding`, passing in the email text:

<Spoiler>

```typescript
for (const email of emails) {
  const cachedEmbedding = await getCachedEmbedding(
    emailToText(email),
  );
  if (cachedEmbedding) {
    results.push({ id: email.id, embedding: cachedEmbedding });
  } else {
    uncachedEmails.push(email);
  }
}
```

</Spoiler>

- [ ] Update the `embedMany` call in the batch processing loop to use `emailToText`:

<Spoiler>

```typescript
const { embeddings } = await embedMany({
  model: google.textEmbeddingModel('gemini-embedding-001'),
  values: batch.map((e) => emailToText(e)),
});
```

</Spoiler>

- [ ] Replace the `fs.writeFile` call with `writeEmbeddingToCache` when writing embeddings to cache:

<Spoiler>

```typescript
for (let j = 0; j < batch.length; j++) {
  const email = batch[j];
  const embedding = embeddings[j];

  await writeEmbeddingToCache(emailToText(email), embedding);

  results.push({ id: email.id, embedding });
}
```

</Spoiler>

## Adding The Text Splitter

<!-- VIDEO -->

Let's add text splitting functionality to chunk emails into smaller pieces for better retrieval.

Recommendation: cherry-pick, and read the code carefully. This is porting material over from the skill-building.

### Steps To Complete

- [ ] Add `@langchain/textsplitters` to your dependencies:

```bash
pnpm add @langchain/textsplitters
```

- [ ] Import `RecursiveCharacterTextSplitter` from `@langchain/textsplitters` in `src/app/search.ts`:

```typescript
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
```

- [ ] Add a new `EmailChunk` type that represents a chunk of an email with its metadata:

<Spoiler>

```typescript
// src/app/search.ts
export type EmailChunk = {
  id: string;
  subject: string;
  chunk: string;
  index: number;
  totalChunks: number;
  from: string;
  to: string | string[];
  timestamp: string;
};
```

</Spoiler>

- [ ] Create a `RecursiveCharacterTextSplitter` instance with appropriate separators for email content. Use a chunk size of 1000 characters and an overlap of 100 characters. Use these separators: `\n\n`, `\n`, ` `, and `''`.

<Spoiler>

```typescript
// src/app/search.ts
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 100,
  separators: ['\n\n', '\n', ' ', ''],
});
```

</Spoiler>

- [ ] Implement the `chunkEmails` function that takes an array of emails and returns chunked email objects:

<Spoiler>

```typescript
// src/app/search.ts
export const chunkEmails = async (emails: Email[]) => {
  const emailsWithChunks: EmailChunk[] = [];
  for (const email of emails) {
    const chunks = await textSplitter.splitText(email.body);

    chunks.forEach((chunk, chunkIndex) => {
      emailsWithChunks.push({
        id: email.id,
        index: chunkIndex,
        subject: email.subject,
        chunk,
        from: email.from,
        to: email.to,
        timestamp: email.timestamp,
        totalChunks: chunks.length,
      });
    });
  }
  return emailsWithChunks;
};
```

</Spoiler>

## Applying Chunking Functions to Search Algorithms

<!-- VIDEO -->

We're now converting all search algorithms from working with whole emails to working with email chunks. This allows for more granular search results.

Recommendation: cherry-pick, and read the code carefully. This refactors our existing functions to work with email chunks. Once this is done, experiment with chunking to see how it affects the search results.

### Steps To Complete

#### Update imports and rename helper function

- [ ] Open `src/app/search.ts` and add the `RecursiveCharacterTextSplitter` import:

```typescript
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
```

- [ ] Rename the `emailToText` function to `emailChunkToText` and update it to work with `EmailChunk`:

<Spoiler>

```typescript
// src/app/search.ts
// CHANGED: emailToText -> emailChunkToText
// CHANGED: Email -> EmailChunk
// CHANGED: email.body -> email.chunk
export const emailChunkToText = (email: EmailChunk) =>
  `${email.subject} ${email.chunk}`;
```

</Spoiler>

#### Update `searchWithBM25` for email chunks

- [ ] Modify the `searchWithBM25` function signature and implementation to accept `EmailChunk[]`:

<Spoiler>

```typescript
// src/app/search.ts
// CHANGED: emails: Email[] -> emailChunks: EmailChunk[]
export async function searchWithBM25(
  keywords: string[],
  emailChunks: EmailChunk[],
) {
  // CHANGED: Use emailChunkToText instead of emailToText
  // CHANGED: Use emailChunks instead of emails
  const corpus = emailChunks.map((emailChunk) =>
    emailChunkToText(emailChunk),
  );

  // BM25 returns score array matching corpus order
  const scores: number[] = (BM25 as any)(corpus, keywords);

  // Map scores to emails, sort descending
  // CHANGED: emails[idx] -> emailChunks[idx]
  return scores
    .map((score, idx) => ({ score, email: emailChunks[idx] }))
    .sort((a, b) => b.score - a.score);
}
```

</Spoiler>

#### Update `loadOrGenerateEmbeddings` for email chunks

- [ ] Update the function signature to accept `EmailChunk[]`:

<Spoiler>

```typescript
// src/app/search.ts
// CHANGED: emails: Email[] -> emailChunks: EmailChunk[]
export async function loadOrGenerateEmbeddings(
  emailChunks: EmailChunk[]
): Promise<{ id: string; embedding: number[] }[]> {
```

</Spoiler>

- [ ] Update the cache-checking loop to use `emailChunks`:

<Spoiler>

```typescript
// src/app/search.ts
const results: { id: string; embedding: number[] }[] = [];
// CHANGED: uncachedEmails -> uncachedEmailChunks
// CHANGED: Email[] -> EmailChunk[]
const uncachedEmailChunks: EmailChunk[] = [];

// CHANGED: for (const email of emails) -> for (const emailChunk of emailChunks)
for (const emailChunk of emailChunks) {
  // CHANGED: Use emailChunkToText instead of emailToText
  const cachedEmbedding = await getCachedEmbedding(
    emailChunkToText(emailChunk),
  );
  if (cachedEmbedding) {
    // CHANGED: email.id -> emailChunk.id
    results.push({
      id: emailChunk.id,
      embedding: cachedEmbedding,
    });
  } else {
    uncachedEmailChunks.push(emailChunk);
  }
}
```

</Spoiler>

- [ ] Update the batch processing loop to use `uncachedEmailChunks`:

<Spoiler>

```typescript
// src/app/search.ts
  // CHANGED: if (uncachedEmails.length > 0) -> if (uncachedEmailChunks.length > 0)
  if (uncachedEmailChunks.length > 0) {
    console.log(
      `Generating embeddings for ${uncachedEmailChunks.length} emails`
    );

    const BATCH_SIZE = 99;
    // CHANGED: i < uncachedEmails.length -> i < uncachedEmailChunks.length
    for (let i = 0; i < uncachedEmailChunks.length; i += BATCH_SIZE) {
      // CHANGED: uncachedEmails.slice -> uncachedEmailChunks.slice
      const batch = uncachedEmailChunks.slice(i, i + BATCH_SIZE);
      console.log(
        `Processing batch ${Math.floor(i / BATCH_SIZE) + 1}/${Math.ceil(
          // CHANGED: uncachedEmails.length -> uncachedEmailChunks.length
          uncachedEmailChunks.length / BATCH_SIZE
        )}`
      );

      const { embeddings } = await embedMany({
        model: google.textEmbeddingModel("text-embedding-004"),
        // CHANGED: batch.map((e) => emailToText(e)) -> batch.map((e) => emailChunkToText(e))
        values: batch.map((e) => emailChunkToText(e)),
      });

      // Write batch to cache
      for (let j = 0; j < batch.length; j++) {
        const email = batch[j];
        const embedding = embeddings[j];

        // CHANGED: emailToText -> emailChunkToText
        await writeEmbeddingToCache(emailChunkToText(email), embedding);

        results.push({ id: email.id, embedding });
      }
    }
  }

  return results;
}
```

</Spoiler>

#### Update `searchWithEmbeddings` for email chunks

- [ ] Update the function signature to accept `EmailChunk[]`:

<Spoiler>

```typescript
// src/app/search.ts
// CHANGED: emails: Email[] -> emailChunks: EmailChunk[]
export async function searchWithEmbeddings(
  query: string,
  emailChunks: EmailChunk[]
) {
```

</Spoiler>

- [ ] Update the embeddings loading and email lookup:

<Spoiler>

```typescript
// src/app/search.ts
  // Load cached embeddings
  // CHANGED: emails -> emailChunks
  const emailEmbeddings = await loadOrGenerateEmbeddings(emailChunks);

  // Generate query embedding
  const { embedding: queryEmbedding } = await embed({
    model: google.textEmbeddingModel("text-embedding-004"),
    value: query,
  });

  // Calculate similarity scores
  // CHANGED: const email = emails.find -> const email = emailChunks.find
  const results = emailEmbeddings.map(({ id, embedding }) => {
    const email = emailChunks.find((e) => e.id === id)!;
    const score = cosineSimilarity(queryEmbedding, embedding);
    return { score, email };
  });

  // Sort by similarity descending
  return results.sort((a, b) => b.score - a.score);
}
```

</Spoiler>

#### Update `reciprocalRankFusion` for email chunks

- [ ] Update the type signatures and scoring loop:

<Spoiler>

```typescript
// src/app/search.ts
// CHANGED: email: Email -> email: EmailChunk (in both type signatures)
export function reciprocalRankFusion(
  rankings: { email: EmailChunk; score: number }[][],
): { email: EmailChunk; score: number }[] {
  const rrfScores = new Map<string, number>();
  // CHANGED: Map<string, Email> -> Map<string, EmailChunk>
  const emailMap = new Map<string, EmailChunk>();

  // Process each ranking list (BM25 and embeddings)
  rankings.forEach((ranking) => {
    ranking.forEach((item, rank) => {
      // CHANGED: Create unique key combining email ID and chunk index
      const emailChunkId = `${item.email.id}-${item.email.index}`;

      const currentScore = rrfScores.get(emailChunkId) || 0;

      // Position-based scoring: 1/(k+rank)
      const contribution = 1 / (RRF_K + rank);
      // CHANGED: Use emailChunkId instead of item.email.id
      rrfScores.set(emailChunkId, currentScore + contribution);

      emailMap.set(emailChunkId, item.email);
    });
  });

  // Sort by combined RRF score descending
  return (
    Array.from(rrfScores.entries())
      .sort(([, scoreA], [, scoreB]) => scoreB - scoreA)
      // CHANGED: Use emailChunkId instead of emailId
      .map(([emailChunkId, score]) => ({
        score,
        email: emailMap.get(emailChunkId)!,
      }))
  );
}
```

</Spoiler>

#### Update `searchWithRRF` to use email chunks

- [ ] Update the function to chunk emails and pass them to search functions:

<Spoiler>

```typescript
// src/app/search.ts
export const searchWithRRF = async (
  query: string,
  emails: Email[],
) => {
  // ADDED: Chunk emails before searching
  const emailChunks = await chunkEmails(emails);
  // CHANGED: Pass emailChunks instead of emails
  const bm25Ranking = await searchWithBM25(
    query.toLowerCase().split(' '),
    emailChunks,
  );
  // CHANGED: Pass emailChunks instead of emails
  const embeddingsRanking = await searchWithEmbeddings(
    query,
    emailChunks,
  );
  const rrfRanking = reciprocalRankFusion([
    bm25Ranking,
    embeddingsRanking,
  ]);
  return rrfRanking;
};
```

</Spoiler>

#### Update search tool to use email chunks

- [ ] Open `src/app/api/chat/search-tool.ts` and add `chunkEmails` to the imports:

```typescript
import {
  chunkEmails,
  loadEmails,
  reciprocalRankFusion,
  searchWithBM25,
  searchWithEmbeddings,
} from '@/app/search';
```

- [ ] Add email chunking and update the search function calls:

<Spoiler>

```typescript
// src/app/api/chat/search-tool.ts

const emails = await loadEmails();
// ADDED: Chunk emails into smaller pieces
const emailChunks = await chunkEmails(emails);

// Use search algorithm from lesson 2.2
// CHANGED: Pass emailChunks instead of emails
const bm25Results = keywords
  ? await searchWithBM25(keywords, emailChunks)
  : [];
// CHANGED: Pass emailChunks instead of emails
const embeddingResults = searchQuery
  ? await searchWithEmbeddings(searchQuery, emailChunks)
  : [];
```

</Spoiler>

- [ ] Update the result mapping to use chunk data:

<Spoiler>

```typescript
// src/app/api/chat/search-tool.ts

// Return top 10 full email objects
const topEmails = rrfResults
  .slice(0, 10)
  .filter((r) => r.score > 0) // Only return emails with a score greater than 0
  .map((r) => ({
    id: r.email.id,
    subject: r.email.subject,
    // CHANGED: email.body -> email.chunk (chunk instead of full email body)
    body: r.email.chunk,
    score: r.score,
  }));
```

</Spoiler>

#### Update email list component for chunks

- [ ] Open `src/app/search/email-list.tsx` and update the `Email` type:

<Spoiler>

```typescript
// src/app/search/email-list.tsx

type Email = {
  id: string;
  from: string;
  subject: string;
  preview: string;
  content: string;
  date: string;
  // ADDED: Track which chunk this is
  chunkIndex: number;
  // ADDED: Track total chunks for this email
  totalChunks: number;
};
```

</Spoiler>

- [ ] Update the `EmailCard` heading to display chunk information:

<Spoiler>

```typescript
// src/app/search/email-list.tsx

// CHANGED: Added chunk information to the heading
<h3 className="font-semibold text-base mb-0.5">
  {email.subject} (Chunk {email.chunkIndex + 1} of{" "}
  {email.totalChunks})
</h3>
```

</Spoiler>

#### Update search page to extract chunk data

- [ ] Open `src/app/search/page.tsx` and update the email transformation:

<Spoiler>

```typescript
// src/app/search/page.tsx

const transformedEmails = emailsWithScores
  .map(({ email, score }) => ({
    id: email.id,
    from: email.from,
    subject: email.subject,
    // CHANGED: email.body -> email.chunk
    preview: email.chunk.substring(0, 100) + '...',
    // CHANGED: email.body -> email.chunk
    content: email.chunk,
    // ADDED: Extract chunk index
    chunkIndex: email.index,
    // ADDED: Extract total chunks
    totalChunks: email.totalChunks,
    date: email.timestamp,
    score: score,
  }))
  .sort((a, b) => b.score - a.score);
```

</Spoiler>

#### Test the changes

- [ ] Run the development server to verify the search works with chunks:

```bash
pnpm dev
```

- [ ] Search for a query and verify that results display with chunk information (e.g., "Chunk 1 of 8").
