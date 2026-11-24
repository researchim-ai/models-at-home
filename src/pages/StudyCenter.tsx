import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { BookOpen, FileText, RefreshCw } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { api } from '@/api/client'

interface DocEntry {
  name: string
  path: string
  source: 'local' | 'github'
}

export function StudyCenter() {
  const { t, i18n } = useTranslation()
  const [docs, setDocs] = useState<DocEntry[]>([])
  const [selectedDoc, setSelectedDoc] = useState<string | null>(null)
  const [content, setContent] = useState<string>('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    loadIndex()
  }, [i18n.language])

  const loadIndex = async () => {
    try {
      const data = await api.studyIndex(i18n.language)
      setDocs(data.documents ?? [])
    } catch { /* offline */ }
  }

  const loadDocument = async (path: string) => {
    setLoading(true)
    try {
      const data = await api.studyDocument(path)
      setContent(data.content ?? '')
      setSelectedDoc(path)
    } catch {
      setContent('Failed to load document')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex h-full">
      {/* Sidebar with document list */}
      <div className="w-72 shrink-0 border-r border-border p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            {t('study.title')}
          </h3>
          <Button variant="ghost" size="icon" onClick={loadIndex}>
            <RefreshCw className="h-3.5 w-3.5" />
          </Button>
        </div>
        <ScrollArea className="h-[calc(100vh-180px)]">
          <div className="space-y-1">
            {docs.map((doc) => (
              <button
                key={doc.path}
                onClick={() => loadDocument(doc.path)}
                className={`flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm transition-colors text-left ${
                  selectedDoc === doc.path
                    ? 'bg-accent text-accent-foreground'
                    : 'text-muted-foreground hover:bg-accent/50 hover:text-foreground'
                }`}
              >
                <FileText className="h-3.5 w-3.5 shrink-0" />
                <span className="truncate">{doc.name}</span>
              </button>
            ))}
            {docs.length === 0 && (
              <div className="text-center text-sm text-muted-foreground py-8">
                <BookOpen className="mx-auto h-8 w-8 mb-2 opacity-30" />
                <p>No documents found</p>
              </div>
            )}
          </div>
        </ScrollArea>
      </div>

      {/* Document content */}
      <ScrollArea className="flex-1 p-8">
        {loading ? (
          <div className="text-muted-foreground text-sm">{t('common.loading')}</div>
        ) : content ? (
          <article className="prose prose-invert max-w-4xl mx-auto">
            <ReactMarkdown>{content}</ReactMarkdown>
          </article>
        ) : (
          <div className="flex h-96 items-center justify-center text-muted-foreground">
            <div className="text-center">
              <BookOpen className="mx-auto h-12 w-12 mb-3 opacity-20" />
              <p className="text-sm">{t('study.subtitle')}</p>
              <p className="text-xs mt-1">Select a document from the sidebar</p>
            </div>
          </div>
        )}
      </ScrollArea>
    </div>
  )
}
