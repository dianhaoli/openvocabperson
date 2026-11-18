import { api } from '../../api/client';
import { useApi } from '../../hooks/useApi';
import type { Person } from '../../types';
import { Img } from './Img';

const cache = new Map<string, Promise<Person>>();

function loadPerson(id: string): Promise<Person> {
  let p = cache.get(id);
  if (!p) {
    p = api.person(id);
    p.catch(() => cache.delete(id));
    cache.set(id, p);
  }
  return p;
}

/** Thumbnail for a person known only by id (fetches the Person once, cached). */
export function PersonThumb({ personId, className }: { personId: string; className?: string }) {
  const { data } = useApi(`person-thumb:${personId}`, () => loadPerson(personId));
  return <Img src={data?.thumb_url ?? undefined} alt="person" className={className} />;
}
