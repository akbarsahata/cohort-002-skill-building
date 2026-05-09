Right now, the search page in your app has a very basic search algorithm. It just checks if the `subject`, `from`, or `content` includes the query text. No ranking, no intelligence, nothing fancy.

But search is one of the most important features in any application. A dumb search makes users frustrated. A smart search makes them happy.

Over the next few exercises, you're going to transform this basic search into something powerful. You'll implement BM25, embeddings, and re-ranking to make the search truly intelligent.

## Adding BM25 Search

<!-- VIDEO -->

Let's start by adding BM25 search to the search page.

Recommendation: cherry-pick. This commit sets up the search pipeline for the project.

### Steps To Complete

#### Adding the `okapibm25` package

- [ ] Add the `okapibm25` package to the project

```bash
pnpm add okapibm25
```

#### Creating the `searchWithBM25` function

- [ ] Create a new file called `search.ts` in the `src/app` directory with a `searchWithBM25` function that takes keywords and emails.

<Spoiler>

```typescript
// src/app/search.ts
// ADDED: New function to search emails using BM25 algorithm
export async function searchWithBM25(
  keywords: string[],
  emails: Email[],
) {
  // Combine subject + body so BM25 searches across both fields
  const corpus = emails.map(
    (email) => `${email.subject} ${email.body}`,
  );

  // BM25 returns score array matching corpus order
  const scores: number[] = (BM25 as any)(corpus, keywords);

  // Map scores to emails, sort descending
  return scores
    .map((score, idx) => ({ score, email: emails[idx] }))
    .sort((a, b) => b.score - a.score);
}
```

</Spoiler>

#### Colocating the Search Functionality

- [ ] Move the existing `loadEmails` function and `Email` interface to the `search.ts` file to keep all search-related code together.

<Spoiler>

```typescript
// src/app/search.ts
// ADDED: Moved from search page for colocation
export async function loadEmails(): Promise<Email[]> {
  const filePath = path.join(
    process.cwd(),
    'data',
    'emails.json',
  );
  const fileContent = await fs.readFile(filePath, 'utf-8');
  return JSON.parse(fileContent);
}

// ADDED: Moved from search page for colocation
export interface Email {
  id: string;
  threadId: string;
  from: string;
  to: string | string[];
  cc?: string[];
  subject: string;
  body: string;
  timestamp: string;
  inReplyTo?: string;
  references?: string[];
  labels?: string[];
  arcId?: string;
  phaseId?: number;
}
```

</Spoiler>

#### Updating the Search Page

- [ ] Import the `loadEmails` and `searchWithBM25` functions in `src/app/search/page.tsx`.

```typescript
// src/app/search/page.tsx
// ADDED: Import new search functions
import { loadEmails, searchWithBM25 } from '../search';
```

- [ ] Replace the old search logic with `searchWithBM25`. This switches to BM25 ranking instead of basic filtering.

<Spoiler>

```typescript
// src/app/search/page.tsx
// CHANGED: Use BM25 search instead of previous filtering logic
const emailsWithScores = await searchWithBM25(
  query.toLowerCase().split(' '),
  allEmails,
);
```

</Spoiler>

- [ ] Update `transformedEmails` to map from `emailsWithScores` and include the score.

<Spoiler>

```typescript
// CHANGED: Map from emailsWithScores instead of allEmails
const transformedEmails = emailsWithScores.map(
  ({ email, score }) => ({
    id: email.id,
    from: email.from,
    subject: email.subject,
    preview: email.body.substring(0, 100) + '...',
    content: email.body,
    date: email.timestamp,
    score: score, // ADDED: Include BM25 score
  }),
);
```

</Spoiler>

- [ ] Sort emails by BM25 score instead of date.

<Spoiler>

```typescript
// CHANGED: Sort by BM25 score descending instead of date
const transformedEmails = emailsWithScores
  .map(({ email, score }) => ({
    id: email.id,
    from: email.from,
    subject: email.subject,
    preview: email.body.substring(0, 100) + '...',
    content: email.body,
    date: email.timestamp,
    score: score,
  }))
  .sort((a, b) => b.score - a.score);
```

</Spoiler>

- [ ] Filter emails by score instead of string matching to exclude irrelevant results.

<Spoiler>

```typescript
// CHANGED: Filter by BM25 score instead of string matching
const filteredEmails = query
  ? transformedEmails.filter((email) => email.score > 0)
  : transformedEmails;
```

</Spoiler>

#### Testing

- [ ] Run the development server and search for a query.

```bash
# Terminal
pnpm dev
```

- [ ] You should see search results sorted by BM25 score in descending order.

## Caching Embeddings

<!-- VIDEO -->

Let's implement a system for generating and caching embeddings so we don't regenerate them unnecessarily.

Recommendation: cherry-pick, and read the code carefully. Caching embeddings is a common pattern in AI applications, so it's worth understanding.

### Steps To Complete

#### Adding embedding imports

- [ ] Add imports to `src/app/search.ts` for the embedding functionality.

```typescript
// src/app/search.ts
// ADDED: Import embedding functions
import { embedMany } from 'ai';
import { google } from '@ai-sdk/google';
```

#### Setting up cache configuration

- [ ] Add cache configuration constants and a helper function to get the embedding file path.

<Spoiler>

```typescript
// src/app/search.ts
// ADDED: Cache configuration for embeddings
const CACHE_DIR = path.join(process.cwd(), 'data', 'embeddings');

const CACHE_KEY = 'google-text-embedding-004';

const getEmbeddingFilePath = (id: string) =>
  path.join(CACHE_DIR, `${CACHE_KEY}-${id}.json`);
```

</Spoiler>

#### Implementing `loadOrGenerateEmbeddings`

- [ ] Create a function that loads cached embeddings or generates new ones in batches.

<Spoiler>

```typescript
// src/app/search.ts
// ADDED: Load embeddings from cache or generate new ones
export async function loadOrGenerateEmbeddings(
  emails: Email[],
): Promise<{ id: string; embedding: number[] }[]> {
  // Ensure cache directory exists
  await fs.mkdir(CACHE_DIR, { recursive: true });

  const results: { id: string; embedding: number[] }[] = [];
  const uncachedEmails: Email[] = [];

  // Check cache for each email
  for (const email of emails) {
    try {
      const cached = await fs.readFile(
        getEmbeddingFilePath(email.id),
        'utf-8',
      );
      const data = JSON.parse(cached);
      results.push({ id: email.id, embedding: data.embedding });
    } catch {
      // Cache miss - need to generate
      uncachedEmails.push(email);
    }
  }

  // Generate embeddings for uncached emails in batches of 99
  if (uncachedEmails.length > 0) {
    console.log(
      `Generating embeddings for ${uncachedEmails.length} emails`,
    );

    const BATCH_SIZE = 99;
    for (let i = 0; i < uncachedEmails.length; i += BATCH_SIZE) {
      const batch = uncachedEmails.slice(i, i + BATCH_SIZE);
      console.log(
        `Processing batch ${Math.floor(i / BATCH_SIZE) + 1}/${Math.ceil(
          uncachedEmails.length / BATCH_SIZE,
        )}`,
      );

      const { embeddings } = await embedMany({
        model: google.textEmbeddingModel('gemini-embedding-001'),
        values: batch.map((e) => `${e.subject} ${e.body}`),
      });

      // Write batch to cache
      for (let j = 0; j < batch.length; j++) {
        const email = batch[j];
        const embedding = embeddings[j];

        await fs.writeFile(
          getEmbeddingFilePath(email.id),
          JSON.stringify({ id: email.id, embedding }),
        );

        results.push({ id: email.id, embedding });
      }
    }
  }

  return results;
}
```

</Spoiler>

#### Calling embeddings on page load

- [ ] Update imports in `src/app/search/page.tsx` to include `loadOrGenerateEmbeddings`.

```typescript
// src/app/search/page.tsx
// ADDED: Import embedding function
import {
  loadEmails,
  loadOrGenerateEmbeddings,
  searchWithBM25,
} from '../search';
```

- [ ] Call `loadOrGenerateEmbeddings` on page load to generate and cache embeddings.

<Spoiler>

```typescript
// src/app/search/page.tsx
// ADDED: Generate and cache embeddings on page load
const allEmails = await loadEmails();

const embeddings = await loadOrGenerateEmbeddings(allEmails);

console.log('Email embeddings loaded:', embeddings.length);
```

</Spoiler>

#### Testing

- [ ] Run the development server and navigate to `/search`.

```bash
# Terminal
pnpm dev
```

- [ ] On first load, you'll see console logs showing batch progress as embeddings are generated.

- [ ] Refresh the page—embeddings should load from cache instantly without regenerating.

## Searching with Embeddings

<!-- VIDEO -->

Let's implement semantic search using embeddings and cosine similarity.

Recommendation: code this commit by hand to make sure you've understood the concepts.

### Steps To Complete

#### Updating imports in `search.ts`

- [ ] Update the imports in `src/app/search.ts` to include `embed` and `cosineSimilarity`.

```typescript
// src/app/search.ts
// CHANGED: Add new AI functions
import { embed, embedMany, cosineSimilarity } from 'ai';
```

#### Creating `searchWithEmbeddings`

- [ ] Add a new function to search emails using semantic similarity. It should take a query and a list of emails, and return a list of emails sorted by similarity to the query.

<Spoiler>

```typescript
// src/app/search.ts
// ADDED: Semantic search using embeddings and cosine similarity
export async function searchWithEmbeddings(
  query: string,
  emails: Email[],
) {
  // Load cached embeddings
  const emailEmbeddings = await loadOrGenerateEmbeddings(emails);

  // Generate query embedding
  const { embedding: queryEmbedding } = await embed({
    model: google.textEmbeddingModel('gemini-embedding-001'),
    value: query,
  });

  // Calculate similarity scores
  const results = emailEmbeddings.map(({ id, embedding }) => {
    const email = emails.find((e) => e.id === id)!;
    const score = cosineSimilarity(queryEmbedding, embedding);
    return { score, email };
  });

  console.log('Results:', results.length);

  // Sort by similarity descending
  return results.sort((a, b) => b.score - a.score);
}
```

</Spoiler>

#### Updating the search page

- [ ] Update the import in `src/app/search/page.tsx` to use `searchWithEmbeddings` instead of `searchWithBM25`.

```typescript
// src/app/search/page.tsx
// CHANGED: Import embeddings search instead of BM25
import { loadEmails, searchWithEmbeddings } from '../search';
```

- [ ] Replace the BM25 search call with `searchWithEmbeddings`.

<Spoiler>

```typescript
// src/app/search/page.tsx
// CHANGED: Use embeddings search instead of BM25
const emailsWithScores = await searchWithEmbeddings(
  query,
  allEmails,
);
```

</Spoiler>

#### Testing semantic search

- [ ] Run the development server and test with semantic queries like "really nice email about climbing".

```bash
# Terminal
pnpm dev
```

- [ ] You should see emails ranked by semantic relevance, even without exact keyword matches.

## Combining Results with Reciprocal Rank Fusion

<!-- VIDEO -->

Let's implement Reciprocal Rank Fusion (RRF), a rank fusion algorithm that combines BM25 and embeddings results using position-based scoring.

Recommendation: hand-code this commit. This commit applies RRF to the search pipeline, so it brings it all together.

### Steps To Complete

#### Creating the RRF function

- [ ] In `src/app/search.ts`, add the RRF constant and implement the fusion function.

<Spoiler>

```typescript
// src/app/search.ts
// ADDED: RRF parameter for rank fusion
const RRF_K = 60;

// ADDED: Combines multiple ranking lists using position-based scoring
export function reciprocalRankFusion(
  rankings: { email: Email; score: number }[][],
): { email: Email; score: number }[] {
  const rrfScores = new Map<string, number>();
  const emailMap = new Map<string, Email>();

  // Process each ranking list (BM25 and embeddings)
  rankings.forEach((ranking) => {
    ranking.forEach((item, rank) => {
      const currentScore = rrfScores.get(item.email.id) || 0;

      // Position-based scoring: 1/(k+rank)
      const contribution = 1 / (RRF_K + rank);
      rrfScores.set(item.email.id, currentScore + contribution);

      emailMap.set(item.email.id, item.email);
    });
  });

  // Sort by combined RRF score descending
  return Array.from(rrfScores.entries())
    .sort(([, scoreA], [, scoreB]) => scoreB - scoreA)
    .map(([emailId, score]) => ({
      score,
      email: emailMap.get(emailId)!,
    }));
}
```

</Spoiler>

#### Creating `searchWithRRF`

- [ ] Add a function that combines BM25 and embeddings rankings using RRF.

<Spoiler>

```typescript
// src/app/search.ts
// ADDED: Combines BM25 and embeddings search using RRF
export const searchWithRRF = async (
  query: string,
  emails: Email[],
) => {
  const bm25Ranking = await searchWithBM25(
    query.toLowerCase().split(' '),
    emails,
  );
  const embeddingsRanking = await searchWithEmbeddings(
    query,
    emails,
  );
  const rrfRanking = reciprocalRankFusion([
    bm25Ranking,
    embeddingsRanking,
  ]);
  return rrfRanking;
};
```

</Spoiler>

#### Updating the search page

- [ ] Update the import in `src/app/search/page.tsx` to include `searchWithRRF`.

```typescript
// src/app/search/page.tsx
// CHANGED: Import RRF search function
import {
  loadEmails,
  searchWithEmbeddings,
  searchWithRRF,
} from '../search';
```

- [ ] Replace `searchWithEmbeddings` with `searchWithRRF` to use combined results.

<Spoiler>

```typescript
// src/app/search/page.tsx
// CHANGED: Use RRF search combining BM25 and embeddings
const emailsWithScores = await searchWithRRF(query, allEmails);
```

</Spoiler>

#### Testing

- [ ] Run the development server to test your changes.

```bash
# Terminal
pnpm dev
```

- [ ] Navigate to the search page and search for a query like "climbing".

- [ ] Verify that results combine both BM25 and embeddings rankings for improved relevance.
